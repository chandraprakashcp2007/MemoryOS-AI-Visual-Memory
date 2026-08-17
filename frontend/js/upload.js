/**
 * MemoryOS
 * Premium Upload Controller
 *
 * Responsibilities:
 * - File selection
 * - Drag & drop
 * - Client-side validation
 * - Preview generation
 * - Upload queue management
 * - Batch upload
 * - Partial success/failure handling
 * - Retry support
 * - Upload cancellation
 * - Safe DOM rendering
 *
 * Backend communication is handled exclusively through MEMORYOS_API.
 */

"use strict";

/* ==========================================================================
   DEPENDENCY CHECK
   ========================================================================== */

const UPLOAD_CONFIG =
    window.MEMORYOS_CONFIG;

const UPLOAD_API =
    window.MEMORYOS_API;

if (!UPLOAD_CONFIG) {
    throw new Error(
        "MemoryOS configuration is unavailable. " +
        "Load config.js before upload.js."
    );
}

if (!UPLOAD_API) {
    throw new Error(
        "MemoryOS API client is unavailable. " +
        "Load api.js before upload.js."
    );
}

/* ==========================================================================
   DOM SELECTORS
   ========================================================================== */

const UPLOAD_SELECTORS = Object.freeze({
    input: [
        "#fileInput",
        "#uploadInput",
        'input[type="file"]',
    ],

    zone: [
        "#uploadZone",
        "#dropZone",
        ".upload-zone",
        ".upload-dropzone",
    ],

    browseButton: [
        "#browseButton",
        "#browseFiles",
        "[data-action='browse-files']",
    ],

    uploadButton: [
        "#uploadButton",
        "#startUpload",
        "[data-action='start-upload']",
    ],

    clearButton: [
        "#clearUpload",
        "#clearUploads",
        "[data-action='clear-upload']",
    ],

    queue: [
        "#uploadQueue",
        ".upload-queue",
        "[data-upload-queue]",
    ],

    queueCount: [
        "#uploadQueueCount",
        "[data-upload-count]",
    ],

    progress: [
        "#uploadProgress",
        "[data-upload-progress]",
    ],

    progressBar: [
        "#uploadProgressBar",
        "[data-upload-progress-bar]",
    ],

    progressText: [
        "#uploadProgressText",
        "[data-upload-progress-text]",
    ],

    status: [
        "#uploadStatus",
        "[data-upload-status]",
    ],

    emptyState: [
        "#uploadEmpty",
        "[data-upload-empty]",
    ],

    previewGrid: [
        "#uploadPreview",
        "#uploadPreviewGrid",
        ".upload-preview-grid",
        "[data-upload-preview]",
    ],
});

/* ==========================================================================
   DOM UTILITIES
   ========================================================================== */

const firstExistingElement = (selectors) => {
    for (const selector of selectors) {
        const element =
            document.querySelector(selector);

        if (element) {
            return element;
        }
    }

    return null;
};

const allExistingElements = (selectors) => {
    const elements = [];

    for (const selector of selectors) {
        document
            .querySelectorAll(selector)
            .forEach((element) => {
                if (!elements.includes(element)) {
                    elements.push(element);
                }
            });
    }

    return elements;
};

const escapeSelectorValue = (value) => {
    if (
        typeof CSS !== "undefined" &&
        typeof CSS.escape === "function"
    ) {
        return CSS.escape(String(value));
    }

    return String(value).replace(
        /["\\]/g,
        "\\$&"
    );
};

/* ==========================================================================
   FILE ID
   ========================================================================== */

const createFileId = (file) => {
    const randomPart =
        typeof crypto !== "undefined" &&
        typeof crypto.randomUUID === "function"
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random()
                  .toString(36)
                  .slice(2)}`;

    return [
        file.name,
        file.size,
        file.lastModified,
        randomPart,
    ].join("::");
};

/* ==========================================================================
   FILE RECORD
   ========================================================================== */

const createFileRecord = (file) => {
    return {
        id: createFileId(file),

        file,

        name: file.name,

        size: file.size,

        type: file.type,

        lastModified:
            file.lastModified,

        previewUrl: null,

        status: "queued",

        progress: 0,

        error: null,

        response: null,

        memoryId: null,

        duplicate: false,

        retryCount: 0,

        startedAt: null,

        completedAt: null,
    };
};

/* ==========================================================================
   UPLOAD STATUS
   ========================================================================== */

const UPLOAD_STATUS = Object.freeze({
    queued: "queued",

    validating: "validating",

    uploading: "uploading",

    processing: "processing",

    success: "success",

    duplicate: "duplicate",

    failed: "failed",

    cancelled: "cancelled",
});

/* ==========================================================================
   UPLOAD CONTROLLER
   ========================================================================== */

class MemoryOSUploadController {
    constructor() {
        this.files = new Map();

        this.elements = {};

        this.initialized = false;

        this.isUploading = false;

        this.abortController = null;

        this.dragDepth = 0;

        this.maxFiles =
            UPLOAD_CONFIG.upload.maxFiles;

        this.objectUrls = new Set();

        this.bound = false;
    }

    /* ======================================================================
       INITIALIZATION
       ====================================================================== */

    init() {
        if (this.initialized) {
            return this;
        }

        this.cacheElements();

        if (!this.hasUploadInterface()) {
            UPLOAD_CONFIG.logger.warn(
                "Upload interface not found. Upload controller remains available."
            );

            this.initialized = true;

            return this;
        }

        this.bindEvents();

        this.render();

        this.initialized = true;

        UPLOAD_CONFIG.logger.debug(
            "Upload controller initialized."
        );

        return this;
    }

    /* ======================================================================
       CACHE ELEMENTS
       ====================================================================== */

    cacheElements() {
        this.elements.input =
            firstExistingElement(
                UPLOAD_SELECTORS.input
            );

        this.elements.zone =
            firstExistingElement(
                UPLOAD_SELECTORS.zone
            );

        this.elements.browseButton =
            firstExistingElement(
                UPLOAD_SELECTORS.browseButton
            );

        this.elements.uploadButton =
            firstExistingElement(
                UPLOAD_SELECTORS.uploadButton
            );

        this.elements.clearButton =
            firstExistingElement(
                UPLOAD_SELECTORS.clearButton
            );

        this.elements.queue =
            firstExistingElement(
                UPLOAD_SELECTORS.queue
            );

        this.elements.queueCount =
            firstExistingElement(
                UPLOAD_SELECTORS.queueCount
            );

        this.elements.progress =
            firstExistingElement(
                UPLOAD_SELECTORS.progress
            );

        this.elements.progressBar =
            firstExistingElement(
                UPLOAD_SELECTORS.progressBar
            );

        this.elements.progressText =
            firstExistingElement(
                UPLOAD_SELECTORS.progressText
            );

        this.elements.status =
            firstExistingElement(
                UPLOAD_SELECTORS.status
            );

        this.elements.emptyState =
            firstExistingElement(
                UPLOAD_SELECTORS.emptyState
            );

        this.elements.previewGrid =
            firstExistingElement(
                UPLOAD_SELECTORS.previewGrid
            );
    }

    /* ======================================================================
       INTERFACE CHECK
       ====================================================================== */

    hasUploadInterface() {
        return Boolean(
            this.elements.input ||
                this.elements.zone ||
                this.elements.previewGrid ||
                this.elements.queue
        );
    }

    /* ======================================================================
       EVENTS
       ====================================================================== */

    bindEvents() {
        if (this.bound) {
            return;
        }

        if (this.elements.input) {
            this.elements.input.addEventListener(
                "change",
                (event) => {
                    this.handleFileSelection(
                        event.target.files
                    );

                    event.target.value = "";
                }
            );
        }

        if (this.elements.browseButton) {
            this.elements.browseButton.addEventListener(
                "click",
                (event) => {
                    event.preventDefault();

                    this.openFilePicker();
                }
            );
        }

        if (this.elements.zone) {
            this.bindDropZone(
                this.elements.zone
            );
        }

        if (this.elements.uploadButton) {
            this.elements.uploadButton.addEventListener(
                "click",
                (event) => {
                    event.preventDefault();

                    this.startUpload();
                }
            );
        }

        if (this.elements.clearButton) {
            this.elements.clearButton.addEventListener(
                "click",
                (event) => {
                    event.preventDefault();

                    this.clearQueue();
                }
            );
        }

        document.addEventListener(
            "click",
            (event) => {
                const removeButton =
                    event.target.closest(
                        "[data-upload-remove]"
                    );

                if (removeButton) {
                    const id =
                        removeButton.getAttribute(
                            "data-upload-remove"
                        );

                    if (id) {
                        this.removeFile(id);
                    }

                    return;
                }

                const retryButton =
                    event.target.closest(
                        "[data-upload-retry]"
                    );

                if (retryButton) {
                    const id =
                        retryButton.getAttribute(
                            "data-upload-retry"
                        );

                    if (id) {
                        this.retryFile(id);
                    }

                    return;
                }
            }
        );

        this.bound = true;
    }

    /* ======================================================================
       DROP ZONE
       ====================================================================== */

    bindDropZone(zone) {
        zone.addEventListener(
            "dragenter",
            (event) => {
                if (
                    !this.hasFilesInDataTransfer(
                        event.dataTransfer
                    )
                ) {
                    return;
                }

                event.preventDefault();

                this.dragDepth += 1;

                zone.classList.add(
                    "is-dragover"
                );

                zone.setAttribute(
                    "data-drag-active",
                    "true"
                );
            }
        );

        zone.addEventListener(
            "dragover",
            (event) => {
                if (
                    !this.hasFilesInDataTransfer(
                        event.dataTransfer
                    )
                ) {
                    return;
                }

                event.preventDefault();

                if (
                    event.dataTransfer
                ) {
                    event.dataTransfer.dropEffect =
                        "copy";
                }
            }
        );

        zone.addEventListener(
            "dragleave",
            (event) => {
                event.preventDefault();

                this.dragDepth = Math.max(
                    0,
                    this.dragDepth - 1
                );

                if (
                    this.dragDepth === 0
                ) {
                    zone.classList.remove(
                        "is-dragover"
                    );

                    zone.removeAttribute(
                        "data-drag-active"
                    );
                }
            }
        );

        zone.addEventListener(
            "drop",
            (event) => {
                event.preventDefault();

                this.dragDepth = 0;

                zone.classList.remove(
                    "is-dragover"
                );

                zone.removeAttribute(
                    "data-drag-active"
                );

                const files =
                    event.dataTransfer
                        ?.files;

                if (files?.length) {
                    this.handleFileSelection(
                        files
                    );
                }
            }
        );

        zone.addEventListener(
            "click",
            (event) => {
                const interactive =
                    event.target.closest(
                        "button, a, input"
                    );

                if (interactive) {
                    return;
                }

                this.openFilePicker();
            }
        );

        zone.addEventListener(
            "keydown",
            (event) => {
                if (
                    event.key === "Enter" ||
                    event.key === " "
                ) {
                    event.preventDefault();

                    this.openFilePicker();
                }
            }
        );
    }

    /* ======================================================================
       FILE PICKER
       ====================================================================== */

    openFilePicker() {
        if (!this.elements.input) {
            this.notify(
                "Upload input is unavailable.",
                "error"
            );

            return;
        }

        if (this.isUploading) {
            return;
        }

        this.elements.input.click();
    }

    /* ======================================================================
       DATA TRANSFER
       ====================================================================== */

    hasFilesInDataTransfer(
        dataTransfer
    ) {
        if (!dataTransfer) {
            return false;
        }

        if (
            dataTransfer.files &&
            dataTransfer.files.length
        ) {
            return true;
        }

        if (
            dataTransfer.items &&
            dataTransfer.items.length
        ) {
            return Array.from(
                dataTransfer.items
            ).some(
                (item) =>
                    item.kind === "file"
            );
        }

        return false;
    }

    /* ======================================================================
       FILE SELECTION
       ====================================================================== */

    async handleFileSelection(
        fileCollection
    ) {
        if (!fileCollection) {
            return;
        }

        const incomingFiles =
            Array.from(
                fileCollection
            );

        if (
            incomingFiles.length === 0
        ) {
            return;
        }

        const availableSlots =
            this.maxFiles -
            this.files.size;

        if (availableSlots <= 0) {
            this.notify(
                `You can upload up to ${this.maxFiles} files at once.`,
                "warning"
            );

            return;
        }

        const filesToProcess =
            incomingFiles.slice(
                0,
                availableSlots
            );

        if (
            incomingFiles.length >
            availableSlots
        ) {
            this.notify(
                `${availableSlots} files added. The upload limit is ${this.maxFiles}.`,
                "warning"
            );
        }

        for (const file of filesToProcess) {
            await this.addFile(file);
        }

        this.render();
    }

    /* ======================================================================
       ADD FILE
       ====================================================================== */

    async addFile(file) {
        const validation =
            this.validateFile(file);

        if (!validation.valid) {
            this.notify(
                validation.message,
                "error"
            );

            return null;
        }

        const duplicate =
            this.findDuplicateFile(
                file
            );

        if (duplicate) {
            this.notify(
                `${file.name} is already in the upload queue.`,
                "warning"
            );

            return null;
        }

        const record =
            createFileRecord(file);

        record.status =
            UPLOAD_STATUS.validating;

        this.files.set(
            record.id,
            record
        );

        this.render();

        try {
            await this.generatePreview(
                record
            );

            record.status =
                UPLOAD_STATUS.queued;

            record.progress = 0;

            this.render();

            return record;
        } catch (error) {
            record.status =
                UPLOAD_STATUS.failed;

            record.error =
                "Preview could not be generated.";

            this.render();

            UPLOAD_CONFIG.logger.warn(
                "Preview generation failed.",
                error
            );

            return record;
        }
    }

    /* ======================================================================
       VALIDATE FILE
       ====================================================================== */

    validateFile(file) {
        if (!(file instanceof File)) {
            return {
                valid: false,
                message:
                    UPLOAD_CONFIG.errors
                        .invalidFile,
            };
        }

        if (
            !UPLOAD_CONFIG
                .upload
                .acceptedMimeTypes
                .includes(file.type)
        ) {
            return {
                valid: false,

                message:
                    `${file.name}: ` +
                    UPLOAD_CONFIG.errors
                        .invalidFile,
            };
        }

        if (
            file.size >
            UPLOAD_CONFIG
                .upload
                .maxFileSizeBytes
        ) {
            return {
                valid: false,

                message:
                    `${file.name}: ` +
                    UPLOAD_CONFIG.errors
                        .fileTooLarge,
            };
        }

        return {
            valid: true,
            message: null,
        };
    }

    /* ======================================================================
       DUPLICATE DETECTION
       ====================================================================== */

    findDuplicateFile(file) {
        for (const record of this.files.values()) {
            if (
                record.name === file.name &&
                record.size === file.size &&
                record.lastModified ===
                    file.lastModified
            ) {
                return record;
            }
        }

        return null;
    }

    /* ======================================================================
       PREVIEW
       ====================================================================== */

    async generatePreview(record) {
        if (
            !UPLOAD_CONFIG
                .upload
                .generatePreview
        ) {
            return;
        }

        if (
            !record.file.type.startsWith(
                "image/"
            )
        ) {
            return;
        }

        const url =
            URL.createObjectURL(
                record.file
            );

        record.previewUrl = url;

        this.objectUrls.add(url);

        /**
         * Decode the image before considering
         * the preview ready.
         */

        try {
            const image =
                new Image();

            image.decoding =
                "async";

            image.src = url;

            if (
                typeof image.decode ===
                "function"
            ) {
                await image.decode();
            } else {
                await new Promise(
                    (
                        resolve,
                        reject
                    ) => {
                        image.onload =
                            resolve;

                        image.onerror =
                            reject;
                    }
                );
            }
        } catch {
            /**
             * Preview decoding failure should
             * never destroy the upload itself.
             */
        }
    }

    /* ======================================================================
       START UPLOAD
       ====================================================================== */

    async startUpload() {
        if (this.isUploading) {
            return;
        }

        const queuedFiles =
            Array.from(
                this.files.values()
            ).filter(
                (record) =>
                    record.status ===
                        UPLOAD_STATUS.queued ||
                    record.status ===
                        UPLOAD_STATUS.failed ||
                    record.status ===
                        UPLOAD_STATUS.duplicate
            );

        if (
            queuedFiles.length === 0
        ) {
            this.notify(
                "Add at least one image before uploading.",
                "info"
            );

            return;
        }

        this.isUploading = true;

        this.abortController =
            new AbortController();

        this.setUploadingUI(true);

        queuedFiles.forEach(
            (record) => {
                record.status =
                    UPLOAD_STATUS.uploading;

                record.progress = 0;

                record.error = null;

                record.duplicate = false;

                record.startedAt =
                    Date.now();
            }
        );

        this.render();

        try {
            await this.uploadBatch(
                queuedFiles
            );
        } catch (error) {
            if (
                error?.isAbort
            ) {
                queuedFiles.forEach(
                    (record) => {
                        if (
                            record.status ===
                                UPLOAD_STATUS
                                    .uploading ||
                            record.status ===
                                UPLOAD_STATUS
                                    .processing
                        ) {
                            record.status =
                                UPLOAD_STATUS
                                    .cancelled;
                        }
                    }
                );

                this.notify(
                    "Upload cancelled.",
                    "warning"
                );
            } else {
                UPLOAD_CONFIG.logger.error(
                    "Batch upload failed.",
                    error
                );

                queuedFiles.forEach(
                    (record) => {
                        if (
                            record.status ===
                                UPLOAD_STATUS
                                    .uploading
                        ) {
                            record.status =
                                UPLOAD_STATUS
                                    .failed;

                            record.error =
                                UPLOAD_CONFIG
                                    .errors
                                    .uploadFailed;
                        }
                    }
                );

                this.notify(
                    this.getSafeErrorMessage(
                        error,
                        UPLOAD_CONFIG
                            .errors
                            .uploadFailed
                    ),
                    "error"
                );
            }
        } finally {
            this.isUploading = false;

            this.abortController = null;

            this.setUploadingUI(
                false
            );

            this.render();

            this.dispatchUploadEvent();
        }
    }

    /* ======================================================================
       BATCH UPLOAD
       ====================================================================== */

    async uploadBatch(records) {
        const files =
            records.map(
                (record) =>
                    record.file
            );

        /**
         * Simulated local progress.
         *
         * The actual backend request is atomic at
         * HTTP request level, while individual
         * result states are resolved from the response.
         */

        this.setOverallProgress(5);

        const progressTimer =
            window.setInterval(
                () => {
                    const current =
                        this.getOverallProgress();

                    if (
                        current < 88
                    ) {
                        this.setOverallProgress(
                            Math.min(
                                88,
                                current + 4
                            )
                        );
                    }
                },
                500
            );

        try {
            const response =
                await UPLOAD_API.uploadBatch(
                    files,
                    {
                        signal:
                            this.abortController
                                ?.signal,
                    }
                );

            window.clearInterval(
                progressTimer
            );

            this.setOverallProgress(
                92
            );

            this.applyBatchResponse(
                records,
                response
            );

            this.setOverallProgress(
                100
            );
        } catch (error) {
            window.clearInterval(
                progressTimer
            );

            /**
             * If the backend returns an individual
             * response body alongside a non-2xx status,
             * the API client throws. All records remain
             * safely marked as failed rather than
             * crashing the entire frontend.
             */

            records.forEach(
                (record) => {
                    if (
                        record.status ===
                            UPLOAD_STATUS
                                .uploading ||
                        record.status ===
                            UPLOAD_STATUS
                                .processing
                    ) {
                        record.status =
                            UPLOAD_STATUS
                                .failed;

                        record.error =
                            this.getSafeErrorMessage(
                                error,
                                UPLOAD_CONFIG
                                    .errors
                                    .uploadFailed
                            );

                        record.completedAt =
                            Date.now();
                    }
                }
            );

            throw error;
        }
    }

    /* ======================================================================
       APPLY BATCH RESPONSE
       ====================================================================== */

    applyBatchResponse(
        records,
        response
    ) {
        const payload =
            response?.data;

        /**
         * Different backend response shapes can
         * exist. We normalize the common patterns
         * without forcing a new backend contract.
         */

        const resultItems =
            this.extractResultItems(
                payload
            );

        if (
            resultItems.length === 0
        ) {
            /**
             * If the backend returns a successful
             * batch without per-file details,
             * mark the submitted records successful.
             */

            records.forEach(
                (record) => {
                    record.status =
                        UPLOAD_STATUS.success;

                    record.progress = 100;

                    record.completedAt =
                        Date.now();

                    record.response =
                        payload;
                }
            );

            this.notify(
                `${records.length} image${records.length === 1 ? "" : "s"} uploaded successfully.`,
                "success"
            );

            return;
        }

        let successCount = 0;

        let failedCount = 0;

        let duplicateCount = 0;

        resultItems.forEach(
            (item, index) => {
                const record =
                    this.matchResponseToRecord(
                        item,
                        records,
                        index
                    );

                if (!record) {
                    return;
                }

                this.applyResultToRecord(
                    record,
                    item
                );

                if (
                    record.status ===
                    UPLOAD_STATUS.success
                ) {
                    successCount += 1;
                }

                if (
                    record.status ===
                    UPLOAD_STATUS.failed
                ) {
                    failedCount += 1;
                }

                if (
                    record.status ===
                    UPLOAD_STATUS.duplicate
                ) {
                    duplicateCount += 1;
                }
            }
        );

        /**
         * Any records that weren't explicitly
         * returned are treated conservatively.
         */

        records.forEach(
            (record) => {
                if (
                    record.status ===
                    UPLOAD_STATUS.uploading
                ) {
                    record.status =
                        UPLOAD_STATUS.failed;

                    record.progress = 0;

                    record.error =
                        "The backend did not return a result for this file.";
                }
            }
        );

        const parts = [];

        if (successCount) {
            parts.push(
                `${successCount} successful`
            );
        }

        if (duplicateCount) {
            parts.push(
                `${duplicateCount} duplicate`
            );
        }

        if (failedCount) {
            parts.push(
                `${failedCount} failed`
            );
        }

        if (parts.length) {
            this.notify(
                parts.join(" • "),
                failedCount
                    ? "warning"
                    : "success"
            );
        }
    }

    /* ======================================================================
       EXTRACT RESULT ITEMS
       ====================================================================== */

    extractResultItems(payload) {
        if (!payload) {
            return [];
        }

        if (Array.isArray(payload)) {
            return payload;
        }

        const candidates = [
            payload.results,
            payload.items,
            payload.memories,
            payload.uploads,
            payload.files,
            payload.data,
        ];

        for (const candidate of candidates) {
            if (
                Array.isArray(candidate)
            ) {
                return candidate;
            }
        }

        return [];
    }

    /* ======================================================================
       MATCH RESPONSE TO FILE
       ====================================================================== */

    matchResponseToRecord(
        item,
        records,
        index
    ) {
        if (
            !item ||
            typeof item !== "object"
        ) {
            return records[index] || null;
        }

        const responseName =
            item.filename ||
            item.file_name ||
            item.name ||
            item.original_filename;

        if (responseName) {
            const match =
                records.find(
                    (record) =>
                        record.name ===
                        responseName
                );

            if (match) {
                return match;
            }
        }

        const responseId =
            item.id ||
            item.memory_id ||
            item.file_id;

        if (responseId) {
            const match =
                records.find(
                    (record) =>
                        String(
                            record.id
                        ) ===
                        String(
                            responseId
                        )
                );

            if (match) {
                return match;
            }
        }

        return records[index] || null;
    }

    /* ======================================================================
       APPLY RESULT
       ====================================================================== */

    applyResultToRecord(
        record,
        item
    ) {
        record.response = item;

        record.progress = 100;

        record.completedAt =
            Date.now();

        const status =
            String(
                item.status ||
                    item.state ||
                    ""
            ).toLowerCase();

        const isDuplicate =
            Boolean(
                item.duplicate ||
                    status ===
                        "duplicate" ||
                    status ===
                        "already_exists"
            );

        const isFailure =
            Boolean(
                item.error ||
                    status === "failed" ||
                    status === "error"
            );

        if (isDuplicate) {
            record.status =
                UPLOAD_STATUS.duplicate;

            record.duplicate = true;

            record.error =
                item.message ||
                "This memory already exists.";

            return;
        }

        if (isFailure) {
            record.status =
                UPLOAD_STATUS.failed;

            record.error =
                item.error ||
                item.message ||
                UPLOAD_CONFIG
                    .errors
                    .uploadFailed;

            return;
        }

        record.status =
            UPLOAD_STATUS.success;

        record.error = null;

        record.memoryId =
            item.memory_id ||
            item.id ||
            null;
    }

    /* ======================================================================
       RETRY
       ====================================================================== */

    async retryFile(id) {
        const record =
            this.files.get(id);

        if (!record) {
            return;
        }

        if (
            this.isUploading
        ) {
            this.notify(
                "Wait for the current upload to finish.",
                "info"
            );

            return;
        }

        record.status =
            UPLOAD_STATUS.queued;

        record.progress = 0;

        record.error = null;

        record.duplicate = false;

        this.render();

        /**
         * Retry uses the same batch API but with
         * only this individual record.
         */

        this.isUploading = true;

        this.abortController =
            new AbortController();

        this.setUploadingUI(true);

        try {
            record.status =
                UPLOAD_STATUS.uploading;

            this.render();

            await this.uploadBatch([
                record,
            ]);
        } catch (error) {
            record.status =
                UPLOAD_STATUS.failed;

            record.error =
                this.getSafeErrorMessage(
                    error,
                    UPLOAD_CONFIG
                        .errors
                        .uploadFailed
                );
        } finally {
            this.isUploading = false;

            this.abortController = null;

            this.setUploadingUI(
                false
            );

            this.render();

            this.dispatchUploadEvent();
        }
    }

    /* ======================================================================
       CANCEL
       ====================================================================== */

    cancelUpload() {
        if (
            !this.isUploading ||
            !this.abortController
        ) {
            return;
        }

        this.abortController.abort();
    }

    /* ======================================================================
       REMOVE FILE
       ====================================================================== */

    removeFile(id) {
        const record =
            this.files.get(id);

        if (!record) {
            return;
        }

        if (
            record.previewUrl
        ) {
            this.revokeObjectUrl(
                record.previewUrl
            );
        }

        this.files.delete(id);

        this.render();

        this.dispatchUploadEvent();
    }

    /* ======================================================================
       CLEAR QUEUE
       ====================================================================== */

    clearQueue() {
        if (this.isUploading) {
            this.notify(
                "You cannot clear the upload queue while files are uploading.",
                "warning"
            );

            return;
        }

        for (const record of this.files.values()) {
            if (
                record.previewUrl
            ) {
                this.revokeObjectUrl(
                    record.previewUrl
                );
            }
        }

        this.files.clear();

        this.render();

        this.dispatchUploadEvent();
    }

    /* ======================================================================
       REVOKE OBJECT URL
       ====================================================================== */

    revokeObjectUrl(url) {
        if (!url) {
            return;
        }

        try {
            URL.revokeObjectURL(
                url
            );
        } catch {
            // Ignore cleanup failures.
        }

        this.objectUrls.delete(
            url
        );
    }

    /* ======================================================================
       CLEANUP
       ====================================================================== */

    destroy() {
        this.cancelUpload();

        for (const url of this.objectUrls) {
            try {
                URL.revokeObjectURL(
                    url
                );
            } catch {
                // Ignore cleanup failures.
            }
        }

        this.objectUrls.clear();

        this.files.clear();

        this.bound = false;

        this.initialized = false;
    }

    /* ======================================================================
       RENDER
       ====================================================================== */

    render() {
        this.renderQueueCount();

        this.renderPreviewGrid();

        this.renderProgress();

        this.renderStatus();

        this.updateButtons();

        this.renderEmptyState();
    }

    /* ======================================================================
       QUEUE COUNT
       ====================================================================== */

    renderQueueCount() {
        if (
            !this.elements.queueCount
        ) {
            return;
        }

        const count =
            this.files.size;

        this.elements.queueCount.textContent =
            String(count);
    }

    /* ======================================================================
       PREVIEW GRID
       ====================================================================== */

    renderPreviewGrid() {
        const container =
            this.elements.previewGrid ||
            this.elements.queue;

        if (!container) {
            return;
        }

        container.replaceChildren();

        if (
            this.files.size === 0
        ) {
            return;
        }

        const fragment =
            document.createDocumentFragment();

        for (const record of this.files.values()) {
            fragment.appendChild(
                this.createFileCard(
                    record
                )
            );
        }

        container.appendChild(
            fragment
        );
    }

    /* ======================================================================
       FILE CARD
       ====================================================================== */

    createFileCard(record) {
        const article =
            document.createElement(
                "article"
            );

        article.className =
            "upload-file-card";

        article.dataset.uploadFileId =
            record.id;

        article.dataset.status =
            record.status;

        const media =
            document.createElement(
                "div"
            );

        media.className =
            "upload-file-card__media";

        if (record.previewUrl) {
            const image =
                document.createElement(
                    "img"
                );

            image.src =
                record.previewUrl;

            image.alt =
                record.name;

            image.loading =
                "lazy";

            image.decoding =
                "async";

            media.appendChild(
                image
            );
        } else {
            const placeholder =
                document.createElement(
                    "div"
                );

            placeholder.className =
                "upload-file-card__placeholder";

            placeholder.textContent =
                "IMG";

            media.appendChild(
                placeholder
            );
        }

        const body =
            document.createElement(
                "div"
            );

        body.className =
            "upload-file-card__body";

        const name =
            document.createElement(
                "p"
            );

        name.className =
            "upload-file-card__name";

        name.textContent =
            record.name;

        name.title =
            record.name;

        const meta =
            document.createElement(
                "p"
            );

        meta.className =
            "upload-file-card__meta";

        meta.textContent =
            this.formatFileSize(
                record.size
            );

        const status =
            document.createElement(
                "div"
            );

        status.className =
            "upload-file-card__status";

        status.textContent =
            this.getStatusLabel(
                record
            );

        const progress =
            document.createElement(
                "div"
            );

        progress.className =
            "upload-file-card__progress";

        const progressTrack =
            document.createElement(
                "div"
            );

        progressTrack.className =
            "upload-file-card__progress-track";

        const progressFill =
            document.createElement(
                "div"
            );

        progressFill.className =
            "upload-file-card__progress-fill";

        progressFill.style.width =
            `${record.progress}%`;

        progressTrack.appendChild(
            progressFill
        );

        progress.appendChild(
            progressTrack
        );

        body.appendChild(
            name
        );

        body.appendChild(
            meta
        );

        body.appendChild(
            status
        );

        body.appendChild(
            progress
        );

        if (record.error) {
            const error =
                document.createElement(
                    "p"
                );

            error.className =
                "upload-file-card__error";

            error.textContent =
                record.error;

            body.appendChild(
                error
            );
        }

        const actions =
            document.createElement(
                "div"
            );

        actions.className =
            "upload-file-card__actions";

        if (
            record.status ===
                UPLOAD_STATUS.failed ||
            record.status ===
                UPLOAD_STATUS.duplicate
        ) {
            const retry =
                document.createElement(
                    "button"
                );

            retry.type =
                "button";

            retry.className =
                "upload-file-card__retry";

            retry.setAttribute(
                "data-upload-retry",
                record.id
            );

            retry.textContent =
                "Retry";

            actions.appendChild(
                retry
            );
        }

        if (
            record.status !==
                UPLOAD_STATUS.uploading &&
            record.status !==
                UPLOAD_STATUS.processing
        ) {
            const remove =
                document.createElement(
                    "button"
                );

            remove.type =
                "button";

            remove.className =
                "upload-file-card__remove";

            remove.setAttribute(
                "data-upload-remove",
                record.id
            );

            remove.setAttribute(
                "aria-label",
                `Remove ${record.name}`
            );

            remove.textContent =
                "×";

            actions.appendChild(
                remove
            );
        }

        article.appendChild(
            media
        );

        article.appendChild(
            body
        );

        article.appendChild(
            actions
        );

        return article;
    }

    /* ======================================================================
       STATUS LABEL
       ====================================================================== */

    getStatusLabel(record) {
        const labels = {
            queued:
                "Ready to upload",

            validating:
                "Validating…",

            uploading:
                "Uploading…",

            processing:
                "AI processing…",

            success:
                "Indexed successfully",

            duplicate:
                "Already indexed",

            failed:
                "Processing failed",

            cancelled:
                "Cancelled",
        };

        return (
            labels[record.status] ||
            "Ready"
        );
    }

    /* ======================================================================
       PROGRESS
       ====================================================================== */

    renderProgress() {
        const progress =
            this.getOverallProgress();

        this.setOverallProgress(
            progress
        );
    }

    getOverallProgress() {
        if (
            this.files.size === 0
        ) {
            return 0;
        }

        let total = 0;

        for (const record of this.files.values()) {
            total +=
                Number(
                    record.progress
                ) || 0;
        }

        return Math.round(
            total /
                this.files.size
        );
    }

    setOverallProgress(
        progress
    ) {
        const safeProgress =
            Math.min(
                100,
                Math.max(
                    0,
                    Number(progress) || 0
                )
            );

        if (
            this.elements.progressBar
        ) {
            this.elements.progressBar.style.width =
                `${safeProgress}%`;

            this.elements.progressBar.setAttribute(
                "aria-valuenow",
                String(
                    safeProgress
                )
            );
        }

        if (
            this.elements.progressText
        ) {
            this.elements.progressText.textContent =
                `${safeProgress}%`;
        }

        if (
            this.elements.progress
        ) {
            this.elements.progress.dataset.progress =
                String(
                    safeProgress
                );
        }
    }

    /* ======================================================================
       STATUS
       ====================================================================== */

    renderStatus() {
        if (
            !this.elements.status
        ) {
            return;
        }

        const records =
            Array.from(
                this.files.values()
            );

        const success =
            records.filter(
                (record) =>
                    record.status ===
                    UPLOAD_STATUS.success
            ).length;

        const failed =
            records.filter(
                (record) =>
                    record.status ===
                    UPLOAD_STATUS.failed
            ).length;

        const duplicate =
            records.filter(
                (record) =>
                    record.status ===
                    UPLOAD_STATUS.duplicate
            ).length;

        if (
            records.length === 0
        ) {
            this.elements.status.textContent =
                "Ready for your visual memories.";

            return;
        }

        if (this.isUploading) {
            this.elements.status.textContent =
                "MemoryOS is processing your visual memories…";

            return;
        }

        if (
            failed > 0 ||
            duplicate > 0
        ) {
            const parts = [];

            if (success) {
                parts.push(
                    `${success} indexed`
                );
            }

            if (duplicate) {
                parts.push(
                    `${duplicate} duplicate`
                );
            }

            if (failed) {
                parts.push(
                    `${failed} failed`
                );
            }

            this.elements.status.textContent =
                parts.join(" • ");

            return;
        }

        if (
            success ===
            records.length
        ) {
            this.elements.status.textContent =
                "All selected memories are indexed.";
        } else {
            this.elements.status.textContent =
                `${records.length} image${records.length === 1 ? "" : "s"} ready.`;
        }
    }

    /* ======================================================================
       EMPTY STATE
       ====================================================================== */

    renderEmptyState() {
        if (
            !this.elements.emptyState
        ) {
            return;
        }

        const visible =
            this.files.size === 0;

        this.elements.emptyState.hidden =
            !visible;
    }

    /* ======================================================================
       BUTTONS
       ====================================================================== */

    updateButtons() {
        const hasFiles =
            this.files.size > 0;

        if (
            this.elements.uploadButton
        ) {
            this.elements.uploadButton.disabled =
                !hasFiles ||
                this.isUploading;

            this.elements.uploadButton.setAttribute(
                "aria-disabled",
                String(
                    !hasFiles ||
                        this.isUploading
                )
            );
        }

        if (
            this.elements.clearButton
        ) {
            this.elements.clearButton.disabled =
                !hasFiles ||
                this.isUploading;
        }

        if (
            this.elements.browseButton
        ) {
            this.elements.browseButton.disabled =
                this.isUploading;
        }
    }

    /* ======================================================================
       UPLOADING UI
       ====================================================================== */

    setUploadingUI(
        uploading
    ) {
        document.documentElement
            .toggleAttribute(
                "data-uploading",
                uploading
            );

        if (
            this.elements.progress
        ) {
            this.elements.progress.hidden =
                !uploading &&
                this.files.size === 0;
        }

        this.updateButtons();
    }

    /* ======================================================================
       NOTIFICATIONS
       ====================================================================== */

    notify(
        message,
        type = "info"
    ) {
        /**
         * app.js will eventually expose a richer toast
         * system. Until then, dispatch a semantic event.
         */

        document.dispatchEvent(
            new CustomEvent(
                "memoryos:toast",
                {
                    detail: {
                        message,
                        type,
                    },
                }
            )
        );

        UPLOAD_CONFIG.logger.debug(
            "Upload notification",
            {
                message,
                type,
            }
        );
    }

    /* ======================================================================
       SAFE ERROR MESSAGE
       ====================================================================== */

    getSafeErrorMessage(
        error,
        fallback
    ) {
        if (
            error &&
            typeof error.message ===
                "string" &&
            error.message.trim()
        ) {
            return error.message;
        }

        return (
            fallback ||
            UPLOAD_CONFIG.errors
                .unknown
        );
    }

    /* ======================================================================
       FILE SIZE
       ====================================================================== */

    formatFileSize(bytes) {
        if (
            !Number.isFinite(bytes) ||
            bytes <= 0
        ) {
            return "0 B";
        }

        const units = [
            "B",
            "KB",
            "MB",
            "GB",
        ];

        let size = bytes;

        let index = 0;

        while (
            size >= 1024 &&
            index <
                units.length - 1
        ) {
            size /= 1024;

            index += 1;
        }

        return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
    }

    /* ======================================================================
       UPLOAD EVENT
       ====================================================================== */

    dispatchUploadEvent() {
        const records =
            Array.from(
                this.files.values()
            );

        const summary = {
            total:
                records.length,

            successful:
                records.filter(
                    (record) =>
                        record.status ===
                        UPLOAD_STATUS
                            .success
                ).length,

            failed:
                records.filter(
                    (record) =>
                        record.status ===
                        UPLOAD_STATUS
                            .failed
                ).length,

            duplicates:
                records.filter(
                    (record) =>
                        record.status ===
                        UPLOAD_STATUS
                            .duplicate
                ).length,

            pending:
                records.filter(
                    (record) =>
                        record.status ===
                            UPLOAD_STATUS
                                .queued ||
                        record.status ===
                            UPLOAD_STATUS
                                .uploading ||
                        record.status ===
                            UPLOAD_STATUS
                                .processing
                ).length,
        };

        document.dispatchEvent(
            new CustomEvent(
                "memoryos:upload-complete",
                {
                    detail: {
                        summary,
                        records,
                    },
                }
            )
        );
    }

    /* ======================================================================
       PUBLIC STATE
       ====================================================================== */

    getState() {
        return {
            isUploading:
                this.isUploading,

            files:
                Array.from(
                    this.files.values()
                ).map(
                    (record) => ({
                        id:
                            record.id,

                        name:
                            record.name,

                        size:
                            record.size,

                        type:
                            record.type,

                        status:
                            record.status,

                        progress:
                            record.progress,

                        error:
                            record.error,

                        memoryId:
                            record.memoryId,

                        duplicate:
                            record.duplicate,
                    })
                ),

            progress:
                this.getOverallProgress(),
        };
    }
}

/* ==========================================================================
   SINGLETON
   ========================================================================== */

const memoryOSUploader =
    new MemoryOSUploadController();

/* ==========================================================================
   PUBLIC API
   ========================================================================== */

const MEMORYOS_UPLOAD =
    Object.freeze({
        controller:
            memoryOSUploader,

        init:
            () =>
                memoryOSUploader.init(),

        openFilePicker:
            () =>
                memoryOSUploader
                    .openFilePicker(),

        addFile:
            (file) =>
                memoryOSUploader
                    .addFile(file),

        addFiles:
            (files) =>
                memoryOSUploader
                    .handleFileSelection(
                        files
                    ),

        start:
            () =>
                memoryOSUploader
                    .startUpload(),

        cancel:
            () =>
                memoryOSUploader
                    .cancelUpload(),

        retry:
            (id) =>
                memoryOSUploader
                    .retryFile(id),

        remove:
            (id) =>
                memoryOSUploader
                    .removeFile(id),

        clear:
            () =>
                memoryOSUploader
                    .clearQueue(),

        getState:
            () =>
                memoryOSUploader
                    .getState(),
    });

/* ==========================================================================
   GLOBAL EXPOSURE
   ========================================================================== */

window.MEMORYOS_UPLOAD =
    MEMORYOS_UPLOAD;

/* ==========================================================================
   AUTO INITIALIZATION
   ========================================================================== */

const initializeUploadController =
    () => {
        memoryOSUploader.init();
    };

if (
    document.readyState ===
    "loading"
) {
    document.addEventListener(
        "DOMContentLoaded",
        initializeUploadController,
        {
            once: true,
        }
    );
} else {
    initializeUploadController();
}

/* ==========================================================================
   DEVELOPMENT LOG
   ========================================================================== */

UPLOAD_CONFIG.logger.debug(
    "MemoryOS upload module loaded."
);