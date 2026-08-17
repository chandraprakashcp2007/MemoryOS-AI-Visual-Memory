/**
 * MemoryOS — API Client
 * Frontend <-> FastAPI
 */

"use strict";

(() => {
    /* =========================================================
       CONFIG
       ========================================================= */

    if (
        !window.MemoryOS ||
        !window.MemoryOS.config ||
        !window.MemoryOS.utils
    ) {
        console.error(
            "[MemoryOS] config.js must load before api.js"
        );

        return;
    }

    const CONFIG = window.MemoryOS.config;
    const UTILS = window.MemoryOS.utils;


    /* =========================================================
       API ERROR
       ========================================================= */

    class MemoryOSApiError extends Error {
        constructor(message, options = {}) {
            super(message);

            this.name = "MemoryOSApiError";

            this.status =
                options.status ?? null;

            this.code =
                options.code ?? "API_ERROR";

            this.details =
                options.details ?? null;

            this.endpoint =
                options.endpoint ?? null;

            this.method =
                options.method ?? null;

            this.isNetworkError =
                options.isNetworkError ?? false;

            this.isTimeout =
                options.isTimeout ?? false;

            this.isAbort =
                options.isAbort ?? false;
        }
    }


    /* =========================================================
       API CLIENT
       ========================================================= */

    class MemoryOSApiClient {

        constructor() {
            this.activeRequests = new Map();

            this.requestCounter = 0;
        }


        /* =====================================================
           REQUEST ID
           ===================================================== */

        createRequestId() {
            this.requestCounter++;

            return (
                `memoryos-${Date.now()}-${this.requestCounter}`
            );
        }


        /* =====================================================
           URL
           ===================================================== */

        buildUrl(endpoint) {
            return UTILS.buildApiUrl(endpoint);
        }


        /* =====================================================
           REQUEST
           ===================================================== */

        async request(
            endpoint,
            options = {}
        ) {

            const method =
                String(
                    options.method || "GET"
                ).toUpperCase();

            const body =
                options.body ?? null;

            const timeout =
                options.timeout ??
                this.getTimeout(method, endpoint);

            const requestId =
                this.createRequestId();

            const url =
                this.buildUrl(endpoint);

            console.log(
                "[MemoryOS] API REQUEST:",
                method,
                url
            );


            /* -------------------------------------------------
               ABORT / TIMEOUT
               ------------------------------------------------- */

            const controller =
                new AbortController();

            let timeoutId = null;

            if (
                Number.isFinite(timeout) &&
                timeout > 0
            ) {
                timeoutId =
                    window.setTimeout(() => {
                        controller.abort();
                    }, timeout);
            }


            /* -------------------------------------------------
               HEADERS
               ------------------------------------------------- */

            const headers = {
                Accept:
                    "application/json",

                ...(options.headers || {}),
            };


            /*
             * IMPORTANT:
             * Do NOT set Content-Type for FormData.
             */

            if (
                body &&
                !(body instanceof FormData)
            ) {
                headers["Content-Type"] =
                    "application/json";
            }


            /* -------------------------------------------------
               ACTIVE REQUEST
               ------------------------------------------------- */

            this.activeRequests.set(
                requestId,
                {
                    requestId,
                    method,
                    endpoint,
                    startedAt: Date.now(),
                }
            );


            try {

                const fetchOptions = {
                    method,

                    headers,

                    body,

                    signal:
                        controller.signal,
                };


                const response =
                    await fetch(
                        url,
                        fetchOptions
                    );


                if (timeoutId !== null) {
                    clearTimeout(timeoutId);
                }


                /* ---------------------------------------------
                   RESPONSE
                   --------------------------------------------- */

                const contentType =
                    (
                        response.headers.get(
                            "content-type"
                        ) || ""
                    ).toLowerCase();


                let data = null;


                if (
                    contentType.includes(
                        "application/json"
                    )
                ) {

                    try {

                        data =
                            await response.json();

                    } catch {

                        data = null;

                    }

                } else {

                    try {

                        data =
                            await response.text();

                    } catch {

                        data = null;

                    }

                }


                console.log(
                    "[MemoryOS] API RESPONSE:",
                    response.status,
                    url,
                    data
                );


                /* ---------------------------------------------
                   ERROR RESPONSE
                   --------------------------------------------- */

                if (!response.ok) {

                    let message =
                        `Request failed with status ${response.status}.`;


                    if (
                        data &&
                        typeof data === "object"
                    ) {

                        if (
                            typeof data.detail ===
                            "string"
                        ) {
                            message =
                                data.detail;
                        }

                        else if (
                            typeof data.message ===
                            "string"
                        ) {
                            message =
                                data.message;
                        }

                        else if (
                            typeof data.error ===
                            "string"
                        ) {
                            message =
                                data.error;
                        }
                    }


                    throw new MemoryOSApiError(
                        message,
                        {
                            status:
                                response.status,

                            code:
                                `HTTP_${response.status}`,

                            details:
                                data,

                            endpoint,

                            method,
                        }
                    );
                }


                return {
                    ok: true,

                    status:
                        response.status,

                    data,

                    requestId,

                    url,
                };

            } catch (error) {

                if (
                    error instanceof
                    MemoryOSApiError
                ) {
                    throw error;
                }


                if (
                    error &&
                    error.name ===
                    "AbortError"
                ) {

                    throw new MemoryOSApiError(
                        "The request timed out or was cancelled.",
                        {
                            code:
                                "REQUEST_TIMEOUT",

                            endpoint,

                            method,

                            isTimeout:
                                true,
                        }
                    );
                }


                console.error(
                    "[MemoryOS] NETWORK ERROR:",
                    error
                );


                throw new MemoryOSApiError(
                    "Cannot connect to MemoryOS backend. Make sure FastAPI is running on port 8000.",
                    {
                        code:
                            "NETWORK_ERROR",

                        endpoint,

                        method,

                        isNetworkError:
                            true,

                        details:
                            error,
                    }
                );

            } finally {

                if (timeoutId !== null) {
                    clearTimeout(timeoutId);
                }

                this.activeRequests.delete(
                    requestId
                );
            }
        }


        /* =====================================================
           TIMEOUT
           ===================================================== */

        getTimeout(
            method,
            endpoint
        ) {

            if (
                endpoint ===
                CONFIG.api.endpoints.upload
            ) {
                return CONFIG.api.timeout.upload;
            }

            if (
                endpoint ===
                CONFIG.api.endpoints.batchUpload
            ) {
                return CONFIG.api.timeout.batchUpload;
            }

            if (
                endpoint ===
                CONFIG.api.endpoints.search
            ) {
                return CONFIG.api.timeout.search;
            }

            if (
                endpoint ===
                CONFIG.api.endpoints.stats
            ) {
                return CONFIG.api.timeout.stats;
            }

            if (
                endpoint ===
                CONFIG.api.endpoints.health
            ) {
                return CONFIG.api.timeout.health;
            }

            return CONFIG.api.timeout.default;
        }


        /* =====================================================
           GET
           ===================================================== */

        async get(
            endpoint,
            options = {}
        ) {

            return this.request(
                endpoint,
                {
                    ...options,

                    method: "GET",
                }
            );
        }


        /* =====================================================
           POST JSON
           ===================================================== */

        async postJson(
            endpoint,
            data = {},
            options = {}
        ) {

            return this.request(
                endpoint,
                {
                    ...options,

                    method: "POST",

                    body:
                        JSON.stringify(data),

                    headers: {
                        "Content-Type":
                            "application/json",

                        ...(options.headers || {}),
                    },
                }
            );
        }


        /* =====================================================
           UPLOAD IMAGE
           ===================================================== */

        async uploadFile(
            file,
            options = {}
        ) {

            if (!(file instanceof File)) {

                throw new MemoryOSApiError(
                    "Invalid file.",
                    {
                        code:
                            "INVALID_FILE",
                    }
                );
            }


            if (
                !UTILS.isSupportedImageType(
                    file
                )
            ) {

                throw new MemoryOSApiError(
                    "This image type is not supported.",
                    {
                        code:
                            "INVALID_FILE_TYPE",
                    }
                );
            }


            if (
                !UTILS.isFileSizeAllowed(
                    file
                )
            ) {

                throw new MemoryOSApiError(
                    `File is larger than ${CONFIG.upload.maxFileSizeMB} MB.`,
                    {
                        code:
                            "FILE_TOO_LARGE",
                    }
                );
            }


            const formData =
                new FormData();


            formData.append(
                "file",
                file,
                file.name
            );


            console.log(
                "[MemoryOS] Uploading:",
                file.name
            );


            return this.request(
                CONFIG.api.endpoints.upload,
                {
                    method: "POST",

                    body: formData,

                    timeout:
                        options.timeout ??
                        CONFIG.api.timeout.upload,
                }
            );
        }


        /* =====================================================
           BATCH UPLOAD
           ===================================================== */

        async uploadBatch(
            files,
            options = {}
        ) {

            const fileList =
                Array.from(files || []);


            if (
                fileList.length === 0
            ) {

                throw new MemoryOSApiError(
                    "No files selected.",
                    {
                        code:
                            "EMPTY_UPLOAD",
                    }
                );
            }


            if (
                fileList.length >
                CONFIG.upload.maxFilesPerBatch
            ) {

                throw new MemoryOSApiError(
                    `Maximum ${CONFIG.upload.maxFilesPerBatch} files allowed.`,
                    {
                        code:
                            "TOO_MANY_FILES",
                    }
                );
            }


            const formData =
                new FormData();


            for (
                const file
                of fileList
            ) {

                if (
                    !UTILS.isSupportedImageType(
                        file
                    )
                ) {

                    throw new MemoryOSApiError(
                        `${file.name} is not a supported image.`,
                        {
                            code:
                                "INVALID_FILE_TYPE",
                        }
                    );
                }


                if (
                    !UTILS.isFileSizeAllowed(
                        file
                    )
                ) {

                    throw new MemoryOSApiError(
                        `${file.name} is too large.`,
                        {
                            code:
                                "FILE_TOO_LARGE",
                        }
                    );
                }


                formData.append(
                    "files",
                    file,
                    file.name
                );
            }


            return this.request(
                CONFIG.api.endpoints.batchUpload,
                {
                    method: "POST",

                    body: formData,

                    timeout:
                        options.timeout ??
                        CONFIG.api.timeout.batchUpload,
                }
            );
        }


        /* =====================================================
           SEARCH
           ===================================================== */

        async search(
            query,
            options = {}
        ) {

            const normalizedQuery =
                String(
                    query ?? ""
                ).trim();


            if (
                normalizedQuery.length <
                CONFIG.search.minimumQueryLength
            ) {

                throw new MemoryOSApiError(
                    "Please enter a search query.",
                    {
                        code:
                            "EMPTY_SEARCH",
                    }
                );
            }


            if (
                normalizedQuery.length >
                CONFIG.search.maximumQueryLength
            ) {

                throw new MemoryOSApiError(
                    "Search query is too long.",
                    {
                        code:
                            "QUERY_TOO_LONG",
                    }
                );
            }


            const payload = {
                query:
                    normalizedQuery,

                limit:
                    Math.min(
                        options.limit ||
                        CONFIG.search.defaultLimit,

                        CONFIG.search.maximumResults
                    ),
            };


            return this.postJson(
                CONFIG.api.endpoints.search,
                payload,
                {
                    timeout:
                        options.timeout ??
                        CONFIG.api.timeout.search,
                }
            );
        }


        /* =====================================================
           MEMORIES
           ===================================================== */

        async getMemories(
            options = {}
        ) {

            const params =
                new URLSearchParams();


            if (
                options.limit !== undefined
            ) {

                params.set(
                    "limit",
                    String(
                        options.limit
                    )
                );
            }


            if (
                options.offset !== undefined
            ) {

                params.set(
                    "offset",
                    String(
                        options.offset
                    )
                );
            }


            const query =
                params.toString();


            const endpoint =
                query
                    ? `${CONFIG.api.endpoints.memories}?${query}`
                    : CONFIG.api.endpoints.memories;


            return this.get(
                endpoint
            );
        }


        /* =====================================================
           SINGLE MEMORY
           ===================================================== */

        async getMemory(
            memoryId
        ) {

            return this.get(
                CONFIG.api.endpoints
                    .memoryById(memoryId)
            );
        }


        /* =====================================================
           DELETE
           ===================================================== */

        async deleteMemory(
            memoryId
        ) {

            return this.request(
                CONFIG.api.endpoints
                    .deleteMemory(memoryId),
                {
                    method: "DELETE",
                }
            );
        }


        /* =====================================================
           STATS
           ===================================================== */

        async getStats() {

            return this.get(
                CONFIG.api.endpoints.stats
            );
        }


        /* =====================================================
           HEALTH
           ===================================================== */

        async getHealth() {

            return this.get(
                CONFIG.api.endpoints.health
            );
        }


        /* =====================================================
           READINESS
           ===================================================== */

        async getReadiness() {

            return this.get(
                CONFIG.api.endpoints.readiness
            );
        }


        /* =====================================================
           ACTIVE REQUESTS
           ===================================================== */

        getActiveRequests() {

            return Array.from(
                this.activeRequests.values()
            );
        }


        getActiveRequestCount() {

            return this.activeRequests.size;
        }
    }


    /* =========================================================
       CREATE CLIENT
       ========================================================= */

    const client =
        new MemoryOSApiClient();


    /* =========================================================
       GLOBAL API
       ========================================================= */

    window.MEMORYOS_API =
        Object.freeze({

            client,

            MemoryOSApiError,

            request:
                (...args) =>
                    client.request(...args),

            get:
                (...args) =>
                    client.get(...args),

            postJson:
                (...args) =>
                    client.postJson(...args),

            uploadFile:
                (...args) =>
                    client.uploadFile(...args),

            uploadBatch:
                (...args) =>
                    client.uploadBatch(...args),

            search:
                (...args) =>
                    client.search(...args),

            getMemories:
                (...args) =>
                    client.getMemories(...args),

            getMemory:
                (...args) =>
                    client.getMemory(...args),

            deleteMemory:
                (...args) =>
                    client.deleteMemory(...args),

            getStats:
                (...args) =>
                    client.getStats(...args),

            getHealth:
                (...args) =>
                    client.getHealth(...args),

            getReadiness:
                (...args) =>
                    client.getReadiness(...args),

            getActiveRequests:
                () =>
                    client.getActiveRequests(),

            getActiveRequestCount:
                () =>
                    client.getActiveRequestCount(),
        });


    /* =========================================================
       SUCCESS LOG
       ========================================================= */

    console.log(
        "%c[MemoryOS] API client ready",
        "color:#00d4ff;font-weight:bold;"
    );

    console.log(
        "[MemoryOS] Backend:",
        CONFIG.api.baseURL
    );

})();