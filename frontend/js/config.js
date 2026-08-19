/**
 * MemoryOS — Frontend Configuration
 * ----------------------------------
 * Centralized, dependency-free configuration layer.
 *
 * Responsibilities:
 * - Backend URL configuration
 * - API endpoint definitions
 * - Upload constraints
 * - Search configuration
 * - UI timing configuration
 * - Local storage keys
 * - Feature flags
 * - Environment detection
 *
 * IMPORTANT:
 * Never place Gemini keys, Firebase secrets, database credentials,
 * or any private backend secret in this file.
 */

(() => {
    "use strict";

    /* =========================================================
       ENVIRONMENT
       ========================================================= */

    const hostname = window.location.hostname;

    const isLocalhost =
        hostname === "localhost" ||
        hostname === "127.0.0.1" ||
        hostname === "[::1]";

    const isDevelopment =
        isLocalhost ||
        window.location.port !== "";

    const isProduction = !isDevelopment;


    /* =========================================================
       APPLICATION
       ========================================================= */

    const APP = Object.freeze({
        name: "MemoryOS",
        version: "1.0.0",
        environment: isDevelopment ? "development" : "production",

        author: "Chandra Prakash",

        urls: Object.freeze({
            frontend: window.location.origin,
        }),
    });


    /* =========================================================
       BACKEND
       ========================================================= */

    /*
     * Development:
     * FastAPI normally runs on:
     *
     *     http://127.0.0.1:8000
     *
     * Production:
     * Change this to the deployed backend origin.
     *
     * No trailing slash.
     */

    // A non-Vite/legacy build may set this value before loading this file.
    // Never point a deployed visitor at either their own machine or the
    // frontend origin when no API has been configured.
    const DEFAULT_API_BASE_URL = isLocalhost
        ? "http://127.0.0.1:8000"
        : "";

    const API_BASE_URL =
        window.MEMORYOS_API_BASE_URL ||
        DEFAULT_API_BASE_URL;


    /* =========================================================
       API ENDPOINTS
       ========================================================= */

    const API = Object.freeze({
        baseURL: API_BASE_URL,

        endpoints: Object.freeze({
            upload: "/upload",
            batchUpload: "/upload/batch",

            search: "/search",

            memories: "/memories",
            memoryById: (memoryId) =>
                `/memories/${encodeURIComponent(memoryId)}`,

            deleteMemory: (memoryId) =>
                `/memories/${encodeURIComponent(memoryId)}`,

            stats: "/stats",

            health: "/health",
            readiness: "/health/ready",
        }),

        timeout: Object.freeze({
            default: 30000,
            upload: 120000,
            batchUpload: 300000,
            search: 30000,
            stats: 15000,
            health: 10000,
        }),
    });


    /* =========================================================
       UPLOAD CONFIGURATION
       ========================================================= */

    const UPLOAD = Object.freeze({
        maxFilesPerBatch: 20,

        /*
         * Browser-side safety limit.
         * Backend remains the final authority.
         */

        maxFileSizeBytes: 25 * 1024 * 1024,

        maxFileSizeMB: 25,

        acceptedMimeTypes: Object.freeze([
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/gif",
            "image/bmp",
            "image/tiff",
        ]),

        acceptedExtensions: Object.freeze([
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
        ]),

        preview: Object.freeze({
            maxWidth: 1200,
            maxHeight: 1200,
            quality: 0.88,
        }),

        concurrency: 3,

        retry: Object.freeze({
            enabled: true,
            maxAttempts: 2,
            delayMs: 1200,
        }),
    });


    /* =========================================================
       SEARCH CONFIGURATION
       ========================================================= */

    const SEARCH = Object.freeze({
        minimumQueryLength: 1,

        maximumQueryLength: 500,

        debounceMs: 350,

        minimumSearchIntervalMs: 250,

        defaultLimit: 20,

        maximumResults: 100,

        confidence: Object.freeze({
            excellent: 0.90,
            strong: 0.75,
            moderate: 0.55,
            weak: 0.35,
        }),

        suggestions: Object.freeze([
            "the shoes I wanted to buy",
            "my Python study notes",
            "the receipt from my order",
            "travel information I saved",
            "coding screenshots",
            "important documents",
        ]),

        recentSearchesLimit: 8,
    });


    /* =========================================================
       DASHBOARD CONFIGURATION
       ========================================================= */

    const DASHBOARD = Object.freeze({
        refreshIntervalMs: 30000,

        statsTimeoutMs: 15000,

        animationDurationMs: 500,

        numberAnimationDurationMs: 900,

        maxRecentActivityItems: 8,
    });


    /* =========================================================
       UI CONFIGURATION
       ========================================================= */

    const UI = Object.freeze({
        toastDurationMs: 4500,

        toastErrorDurationMs: 6500,

        drawerTransitionMs: 420,

        panelTransitionMs: 300,

        modalTransitionMs: 300,

        commandPaletteTransitionMs: 220,

        tooltipDelayMs: 450,

        loadingMinimumDisplayMs: 350,

        skeletonMinimumDisplayMs: 500,

        notificationStackLimit: 5,
    });


    /* =========================================================
       KEYBOARD SHORTCUTS
       ========================================================= */

    const SHORTCUTS = Object.freeze({
        commandPalette: {
            key: "k",
            modifier: "ctrlOrMeta",
        },

        search: {
            key: "/",
            modifier: null,
        },

        escape: "Escape",
    });


    /* =========================================================
       LOCAL STORAGE
       ========================================================= */

    const STORAGE_KEYS = Object.freeze({
        theme: "memoryos.theme",

        recentSearches: "memoryos.recentSearches",

        lastSearch: "memoryos.lastSearch",

        uploadPreferences: "memoryos.uploadPreferences",

        sidebarState: "memoryos.sidebarState",

        dismissedNotices: "memoryos.dismissedNotices",
    });


    /* =========================================================
       SESSION STORAGE
       ========================================================= */

    const SESSION_KEYS = Object.freeze({
        activeSearch: "memoryos.activeSearch",

        activeMemory: "memoryos.activeMemory",

        uploadSession: "memoryos.uploadSession",

        commandPaletteOpen: "memoryos.commandPaletteOpen",
    });


    /* =========================================================
       FEATURE FLAGS
       ========================================================= */

    const FEATURES = Object.freeze({
        semanticSearch: true,

        hybridSearch: true,

        explainableResults: true,

        batchUpload: true,

        dragAndDrop: true,

        imagePreview: true,

        recentSearches: true,

        searchSuggestions: true,

        commandPalette: true,

        keyboardShortcuts: true,

        analytics: true,

        healthMonitoring: true,

        themeToggle: true,

        memoryDrawer: true,

        filters: true,

        retryFailedUploads: true,

        offlineDetection: true,

        reducedMotionSupport: true,
    });


    /* =========================================================
       MEMORY CARD CONFIGURATION
       ========================================================= */

    const MEMORY_CARD = Object.freeze({
        defaultImageFallback: "assets/images/memory-placeholder.svg",

        maxEntitiesDisplayed: 5,

        maxTagsDisplayed: 6,

        summaryMaximumCharacters: 220,

        titleMaximumCharacters: 100,

        semanticScoreDecimals: 2,

        confidenceDecimals: 0,
    });


    /* =========================================================
       HEALTH STATUS
       ========================================================= */

    const HEALTH = Object.freeze({
        refreshIntervalMs: 30000,

        statuses: Object.freeze({
            healthy: "healthy",
            degraded: "degraded",
            unhealthy: "unhealthy",
            unknown: "unknown",
            loading: "loading",
        }),
    });


    /* =========================================================
       NETWORK
       ========================================================= */

    const NETWORK = Object.freeze({
        retryableStatusCodes: Object.freeze([
            408,
            425,
            429,
            500,
            502,
            503,
            504,
        ]),

        offlineMessage:
            "MemoryOS cannot reach the server. Check your connection and try again.",

        timeoutMessage:
            "The request took too long. Please try again.",

        genericErrorMessage:
            "Something went wrong. Please try again.",
    });


    /* =========================================================
       ACCESSIBILITY
       ========================================================= */

    const ACCESSIBILITY = Object.freeze({
        focusSelector:
            ":focus-visible",

        reducedMotionQuery:
            "(prefers-reduced-motion: reduce)",

        highContrastQuery:
            "(prefers-contrast: more)",

        minimumTouchTargetPx: 44,
    });


    /* =========================================================
       DEBUGGING
       ========================================================= */

    const DEBUG = Object.freeze({
        enabled:
            window.MEMORYOS_DEBUG === true ||
            isDevelopment,

        logApiRequests: isDevelopment,

        logStateChanges: false,

        logPerformance: isDevelopment,
    });


    /* =========================================================
       PERFORMANCE
       ========================================================= */

    const PERFORMANCE = Object.freeze({
        longTaskThresholdMs: 100,

        searchRenderBatchSize: 12,

        imageLazyLoadThreshold: "300px",

        maximumConcurrentImageLoads: 6,
    });


    /* =========================================================
       BUILD INFORMATION
       ========================================================= */

    const BUILD = Object.freeze({
        name: APP.name,

        version: APP.version,

        environment: APP.environment,

        generatedAt: "2026-08-14",

        apiVersion: "v1",
    });


    /* =========================================================
       PUBLIC CONFIG OBJECT
       ========================================================= */

    const CONFIG = Object.freeze({
        app: APP,

        api: API,

        upload: UPLOAD,

        search: SEARCH,

        dashboard: DASHBOARD,

        ui: UI,

        shortcuts: SHORTCUTS,

        storage: STORAGE_KEYS,

        session: SESSION_KEYS,

        features: FEATURES,

        memoryCard: MEMORY_CARD,

        health: HEALTH,

        network: NETWORK,

        accessibility: ACCESSIBILITY,

        debug: DEBUG,

        performance: PERFORMANCE,

        build: BUILD,
    });


    /* =========================================================
       HELPER FUNCTIONS
       ========================================================= */

    /**
     * Build a complete API URL.
     *
     * Example:
     *
     * buildApiUrl("/health")
     * =>
     * http://127.0.0.1:8000/health
     */
    function buildApiUrl(path) {
        if (typeof path !== "string") {
            throw new TypeError("API path must be a string.");
        }

        const normalizedBase =
            API.baseURL.replace(/\/+$/, "");

        const normalizedPath =
            path.startsWith("/")
                ? path
                : `/${path}`;

        return `${normalizedBase}${normalizedPath}`;
    }


    /**
     * Determine whether a file type is supported.
     */
    function isSupportedImageType(file) {
        if (!file) {
            return false;
        }

        return UPLOAD.acceptedMimeTypes.includes(
            file.type.toLowerCase()
        );
    }


    /**
     * Determine whether a file is within the browser-side size limit.
     */
    function isFileSizeAllowed(file) {
        if (!file) {
            return false;
        }

        return file.size <= UPLOAD.maxFileSizeBytes;
    }


    /**
     * Return a human-readable file size.
     */
    function formatBytes(bytes, decimals = 2) {
        if (
            typeof bytes !== "number" ||
            !Number.isFinite(bytes) ||
            bytes < 0
        ) {
            return "0 B";
        }

        if (bytes === 0) {
            return "0 B";
        }

        const units = [
            "B",
            "KB",
            "MB",
            "GB",
            "TB",
        ];

        const base = 1024;

        const exponent = Math.min(
            Math.floor(Math.log(bytes) / Math.log(base)),
            units.length - 1
        );

        const value =
            bytes / Math.pow(base, exponent);

        const precision =
            exponent === 0
                ? 0
                : decimals;

        return `${value.toFixed(precision)} ${units[exponent]}`;
    }


    /**
     * Clamp a numeric value between two boundaries.
     */
    function clamp(value, minimum = 0, maximum = 1) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return minimum;
        }

        return Math.min(
            Math.max(numericValue, minimum),
            maximum
        );
    }


    /**
     * Convert a confidence/score value into a readable percentage.
     *
     * Accepts:
     * 0.93 -> 93%
     * 93   -> 93%
     */
    function formatConfidence(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return "—";
        }

        const percentage =
            numericValue <= 1
                ? numericValue * 100
                : numericValue;

        return `${Math.round(
            clamp(percentage, 0, 100)
        )}%`;
    }


    /**
     * Return confidence level.
     */
    function getConfidenceLevel(value) {
        const numericValue = Number(value);

        if (!Number.isFinite(numericValue)) {
            return "unknown";
        }

        const score =
            numericValue > 1
                ? numericValue / 100
                : numericValue;

        if (score >= SEARCH.confidence.excellent) {
            return "excellent";
        }

        if (score >= SEARCH.confidence.strong) {
            return "strong";
        }

        if (score >= SEARCH.confidence.moderate) {
            return "moderate";
        }

        if (score >= SEARCH.confidence.weak) {
            return "weak";
        }

        return "low";
    }


    /**
     * Safely create a URL for an API endpoint.
     */
    function getEndpoint(name, ...args) {
        const endpoint = API.endpoints[name];

        if (typeof endpoint === "function") {
            return buildApiUrl(endpoint(...args));
        }

        if (typeof endpoint !== "string") {
            throw new Error(
                `Unknown API endpoint: ${name}`
            );
        }

        return buildApiUrl(endpoint);
    }


    /**
     * Detect whether browser is currently offline.
     */
    function isOffline() {
        return navigator.onLine === false;
    }


    /**
     * Return whether reduced motion is requested.
     */
    function prefersReducedMotion() {
        return window.matchMedia &&
            window.matchMedia(
                ACCESSIBILITY.reducedMotionQuery
            ).matches;
    }


    /**
     * Check whether a response status is retryable.
     */
    function isRetryableStatus(status) {
        return NETWORK.retryableStatusCodes.includes(
            Number(status)
        );
    }


    /* =========================================================
       CONFIG UTILITIES
       ========================================================= */

    const CONFIG_UTILS = Object.freeze({
        buildApiUrl,

        getEndpoint,

        isSupportedImageType,

        isFileSizeAllowed,

        formatBytes,

        clamp,

        formatConfidence,

        getConfidenceLevel,

        isOffline,

        prefersReducedMotion,

        isRetryableStatus,
    });


    /* =========================================================
       GLOBAL EXPORT
       ========================================================= */

    /*
     * We expose one immutable global namespace.
     *
     * Later files use:
     *
     *     MemoryOS.config
     *     MemoryOS.utils
     *
     * No framework required.
     */

    window.MemoryOS = window.MemoryOS || {};

    window.MemoryOS.config = CONFIG;

    window.MemoryOS.utils = CONFIG_UTILS;


    /* =========================================================
       DEVELOPMENT DIAGNOSTICS
       ========================================================= */

    if (DEBUG.enabled) {
        console.info(
            `[MemoryOS] ${APP.name} v${APP.version}`
        );

        console.info(
            `[MemoryOS] Environment: ${APP.environment}`
        );

        console.info(
            `[MemoryOS] API: ${API.baseURL}`
        );
    }

})();
