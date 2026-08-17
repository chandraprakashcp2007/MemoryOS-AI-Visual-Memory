/* ==========================================================================
   MemoryOS — Statistics & System Health Controller
   frontend/js/stats.js

   Responsibilities:
   - Fetch /stats
   - Fetch /health
   - Fetch /health/ready
   - Safely normalize backend metrics
   - Populate dashboard statistics
   - Populate processing/search/index health
   - Populate AI/OCR/vector service status
   - Handle missing metrics gracefully
   - Handle backend/network failures
   - Support refresh
   - Dispatch stats lifecycle events
   - Never trust API data blindly
   ========================================================================== */

(() => {
    "use strict";

    /* ----------------------------------------------------------------------
       Namespace
       ---------------------------------------------------------------------- */

    window.MemoryOS = window.MemoryOS || {};

    /* ----------------------------------------------------------------------
       Configuration
       ---------------------------------------------------------------------- */

    const CONFIG = Object.freeze({
        endpoints: Object.freeze({
            stats: "/stats",
            health: "/health",
            ready: "/health/ready",
        }),

        selectors: Object.freeze({
            memories: [
                "#statMemories",
                "[data-stat-memories]",
            ],

            successRate: [
                "#statSuccessRate",
                "[data-stat-success-rate]",
            ],

            searches: [
                "#statSearches",
                "[data-stat-searches]",
            ],

            indexingRate: [
                "#statIndexingRate",
                "[data-stat-indexing-rate]",
            ],

            processingLatency: [
                "#processingLatency",
                "#statProcessingLatency",
                "[data-processing-latency]",
            ],

            searchLatency: [
                "#searchLatency",
                "#statSearchLatency",
                "[data-search-latency]",
            ],

            vectorHealth: [
                "#vectorHealth",
                "#statVectorHealth",
                "[data-vector-health]",
            ],

            ocrHealth: [
                "#ocrHealth",
                "#statOcrHealth",
                "[data-ocr-health]",
            ],

            geminiStatus: [
                "#geminiStatus",
                "#statGeminiStatus",
                "[data-gemini-status]",
            ],

            storage: [
                "#storageMetrics",
                "#statStorage",
                "[data-storage]",
            ],

            categoryMetrics: [
                "#categoryMetrics",
                "[data-category-metrics]",
            ],

            healthStatus: [
                "#systemHealth",
                "#healthStatus",
                "[data-system-health]",
            ],

            readyStatus: [
                "#systemReady",
                "#readyStatus",
                "[data-system-ready]",
            ],

            lastUpdated: [
                "#statsLastUpdated",
                "[data-stats-last-updated]",
            ],

            refreshButtons: [
                "[data-stats-refresh]",
                "#refreshStats",
            ],
        }),

        events: Object.freeze({
            statsLoading: "memoryos:stats-loading",
            statsComplete: "memoryos:stats-complete",
            statsError: "memoryos:stats-error",
            statsRefresh: "memoryos:stats-refresh",
            refresh: "memoryos:refresh",
        }),

        refreshInterval: 60000,

        requestTimeout: 10000,

        fallbackText: "—",
    });

    /* ----------------------------------------------------------------------
       State
       ---------------------------------------------------------------------- */

    const state = {
        initialized: false,
        loading: false,
        error: null,

        stats: null,
        health: null,
        readiness: null,

        lastUpdated: null,

        refreshTimer: null,

        requestController: null,

        requestId: 0,
    };

    /* ----------------------------------------------------------------------
       Generic helpers
       ---------------------------------------------------------------------- */

    function isObject(value) {
        return (
            value !== null &&
            typeof value === "object" &&
            !Array.isArray(value)
        );
    }

    function isArray(value) {
        return Array.isArray(value);
    }

    function firstDefined(...values) {
        for (const value of values) {
            if (
                value !== undefined &&
                value !== null &&
                value !== ""
            ) {
                return value;
            }
        }

        return null;
    }

    function toNumber(value) {
        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            return null;
        }

        const number = Number(value);

        return Number.isFinite(number)
            ? number
            : null;
    }

    function clamp(value, min, max) {
        return Math.min(
            Math.max(value, min),
            max
        );
    }

    function toSafeString(
        value,
        fallback = CONFIG.fallbackText
    ) {
        if (
            value === undefined ||
            value === null
        ) {
            return fallback;
        }

        if (typeof value === "string") {
            const text = value.trim();

            return text || fallback;
        }

        if (
            typeof value === "number" ||
            typeof value === "boolean"
        ) {
            return String(value);
        }

        return fallback;
    }

    function normalizeRatio(value) {
        const number = toNumber(value);

        if (number === null) {
            return null;
        }

        if (number > 1 && number <= 100) {
            return clamp(
                number / 100,
                0,
                1
            );
        }

        return clamp(
            number,
            0,
            1
        );
    }

    function formatNumber(value) {
        const number = toNumber(value);

        if (number === null) {
            return CONFIG.fallbackText;
        }

        try {
            return new Intl.NumberFormat(
                undefined,
                {
                    maximumFractionDigits: 0,
                }
            ).format(number);
        } catch {
            return String(
                Math.round(number)
            );
        }
    }

    function formatPercentage(value) {
        const ratio = normalizeRatio(value);

        if (ratio === null) {
            return CONFIG.fallbackText;
        }

        return `${Math.round(ratio * 100)}%`;
    }

    function formatLatency(value) {
        const number = toNumber(value);

        if (number === null) {
            return CONFIG.fallbackText;
        }

        if (number < 1) {
            return `${Math.round(number * 1000)} ms`;
        }

        if (number < 1000) {
            return `${Math.round(number)} ms`;
        }

        return `${(number / 1000).toFixed(2)} s`;
    }

    function formatBytes(value) {
        const number = toNumber(value);

        if (number === null || number < 0) {
            return CONFIG.fallbackText;
        }

        if (number < 1024) {
            return `${Math.round(number)} B`;
        }

        if (number < 1024 * 1024) {
            return `${(number / 1024).toFixed(1)} KB`;
        }

        if (number < 1024 * 1024 * 1024) {
            return `${(
                number /
                (1024 * 1024)
            ).toFixed(1)} MB`;
        }

        return `${(
            number /
            (1024 * 1024 * 1024)
        ).toFixed(2)} GB`;
    }

    function normalizeStatus(value) {
        if (
            typeof value === "boolean"
        ) {
            return value
                ? "healthy"
                : "unavailable";
        }

        const status =
            toSafeString(
                value,
                "unknown"
            ).toLowerCase();

        if (
            [
                "ok",
                "healthy",
                "ready",
                "online",
                "available",
                "operational",
                "success",
                "up",
            ].includes(status)
        ) {
            return "healthy";
        }

        if (
            [
                "degraded",
                "warning",
                "partial",
                "limited",
            ].includes(status)
        ) {
            return "degraded";
        }

        if (
            [
                "error",
                "failed",
                "failure",
                "offline",
                "unavailable",
                "down",
            ].includes(status)
        ) {
            return "unavailable";
        }

        return "unknown";
    }

    function statusLabel(status) {
        switch (normalizeStatus(status)) {
            case "healthy":
                return "Healthy";

            case "degraded":
                return "Degraded";

            case "unavailable":
                return "Unavailable";

            default:
                return "Unknown";
        }
    }

    /* ----------------------------------------------------------------------
       DOM helpers
       ---------------------------------------------------------------------- */

    function queryFirst(selectors) {
        if (!isArray(selectors)) {
            return null;
        }

        for (const selector of selectors) {
            try {
                const element =
                    document.querySelector(
                        selector
                    );

                if (element) {
                    return element;
                }
            } catch {
                // Optional selector.
            }
        }

        return null;
    }

    function queryAll(selectors) {
        const elements = [];
        const seen = new Set();

        if (!isArray(selectors)) {
            return elements;
        }

        for (const selector of selectors) {
            let matches = [];

            try {
                matches =
                    document.querySelectorAll(
                        selector
                    );
            } catch {
                continue;
            }

            matches.forEach(
                (element) => {
                    if (!seen.has(element)) {
                        seen.add(element);
                        elements.push(element);
                    }
                }
            );
        }

        return elements;
    }

    function setText(
        selectors,
        value
    ) {
        const elements =
            queryAll(selectors);

        elements.forEach(
            (element) => {
                element.textContent =
                    toSafeString(
                        value
                    );
            }
        );
    }

    function setAttribute(
        selectors,
        attribute,
        value
    ) {
        const elements =
            queryAll(selectors);

        elements.forEach(
            (element) => {
                element.setAttribute(
                    attribute,
                    String(value)
                );
            }
        );
    }

    function setDataStatus(
        selectors,
        status
    ) {
        const normalized =
            normalizeStatus(status);

        const elements =
            queryAll(selectors);

        elements.forEach(
            (element) => {
                element.dataset.status =
                    normalized;

                element.setAttribute(
                    "data-health",
                    normalized
                );
            }
        );
    }

    function setVisibility(
        selectors,
        visible
    ) {
        const elements =
            queryAll(selectors);

        elements.forEach(
            (element) => {
                element.hidden =
                    !visible;
            }
        );
    }

    /* ----------------------------------------------------------------------
       Icon support
       ---------------------------------------------------------------------- */

    function refreshIcons(
        root = document
    ) {
        try {
            if (
                window.lucide &&
                typeof window.lucide.createIcons ===
                    "function"
            ) {
                window.lucide.createIcons({
                    root,
                });
            }
        } catch {
            // Icons must never break analytics.
        }
    }

    /* ----------------------------------------------------------------------
       API resolution
       ---------------------------------------------------------------------- */

    function resolveApiClient() {
        if (
            window.MemoryOS &&
            window.MemoryOS.api
        ) {
            return window.MemoryOS.api;
        }

        return null;
    }

    async function request(
        endpoint,
        options = {}
    ) {
        const api =
            resolveApiClient();

        /*
         * Prefer the project's existing API client.
         */

        if (api) {
            const candidates = [
                api.get,
                api.request,
                api.fetch,
            ];

            for (
                const method of candidates
            ) {
                if (
                    typeof method ===
                    "function"
                ) {
                    try {
                        if (
                            method ===
                            api.get
                        ) {
                            return await method.call(
                                api,
                                endpoint,
                                options
                            );
                        }

                        return await method.call(
                            api,
                            endpoint,
                            options
                        );
                    } catch (error) {
                        /*
                         * If the project's client
                         * knows how to handle this
                         * endpoint but rejects it,
                         * propagate the error.
                         */

                        throw error;
                    }
                }
            }
        }

        /*
         * Safe fallback to fetch.
         * No secrets are stored here.
         */

        const controller =
            new AbortController();

        const timeoutId =
            window.setTimeout(
                () => {
                    controller.abort();
                },
                CONFIG.requestTimeout
            );

        try {
            const response =
                await fetch(
                    endpoint,
                    {
                        ...options,
                        signal:
                            controller.signal,
                        headers: {
                            Accept:
                                "application/json",
                            ...(options.headers ||
                                {}),
                        },
                    }
                );

            if (!response.ok) {
                throw new Error(
                    `Request failed with status ${response.status}`
                );
            }

            return await response.json();
        } finally {
            window.clearTimeout(
                timeoutId
            );
        }
    }

    /* ----------------------------------------------------------------------
       Backend response normalization
       ---------------------------------------------------------------------- */

    function unwrapResponse(
        response
    ) {
        if (
            isObject(response) &&
            "data" in response
        ) {
            return response.data;
        }

        return response;
    }

    function extractNumber(
        source,
        paths
    ) {
        if (!isObject(source)) {
            return null;
        }

        for (const path of paths) {
            const parts =
                path.split(".");

            let current =
                source;

            for (
                const part of parts
            ) {
                if (
                    current ===
                        undefined ||
                    current === null
                ) {
                    break;
                }

                current =
                    current[part];
            }

            const number =
                toNumber(current);

            if (number !== null) {
                return number;
            }
        }

        return null;
    }

    function extractValue(
        source,
        paths
    ) {
        if (!isObject(source)) {
            return null;
        }

        for (const path of paths) {
            const parts =
                path.split(".");

            let current =
                source;

            let valid = true;

            for (
                const part of parts
            ) {
                if (
                    current ===
                        undefined ||
                    current === null
                ) {
                    valid = false;
                    break;
                }

                current =
                    current[part];
            }

            if (
                valid &&
                current !==
                    undefined &&
                current !== null &&
                current !== ""
            ) {
                return current;
            }
        }

        return null;
    }

    /* ----------------------------------------------------------------------
       Normalize statistics
       ---------------------------------------------------------------------- */

    function normalizeStats(
        raw
    ) {
        const source =
            unwrapResponse(
                raw
            );

        const memories =
            extractNumber(
                source,
                [
                    "total_memories",
                    "memory_count",
                    "memories",
                    "total",
                    "counts.memories",
                    "counts.total_memories",
                    "data.total_memories",
                    "data.memory_count",
                ]
            );

        const searches =
            extractNumber(
                source,
                [
                    "total_searches",
                    "search_count",
                    "searches",
                    "counts.searches",
                    "counts.total_searches",
                ]
            );

        const successRate =
            extractValue(
                source,
                [
                    "success_rate",
                    "processing_success_rate",
                    "upload_success_rate",
                    "metrics.success_rate",
                    "processing.success_rate",
                ]
            );

        const indexingRate =
            extractValue(
                source,
                [
                    "indexing_rate",
                    "vector_coverage",
                    "vector_coverage_rate",
                    "indexing.coverage",
                    "vector.coverage",
                    "metrics.indexing_rate",
                ]
            );

        const processingLatency =
            extractValue(
                source,
                [
                    "processing_latency",
                    "processing_latency_ms",
                    "average_processing_latency",
                    "avg_processing_latency",
                    "metrics.processing_latency",
                    "processing.average_latency",
                ]
            );

        const searchLatency =
            extractValue(
                source,
                [
                    "search_latency",
                    "search_latency_ms",
                    "average_search_latency",
                    "avg_search_latency",
                    "metrics.search_latency",
                    "search.average_latency",
                ]
            );

        const storage =
            extractValue(
                source,
                [
                    "storage",
                    "storage_bytes",
                    "storage_used",
                    "metrics.storage",
                ]
            );

        const categories =
            extractValue(
                source,
                [
                    "categories",
                    "category_distribution",
                    "category_counts",
                    "metrics.categories",
                ]
            );

        return {
            raw: source,

            memories,

            searches,

            successRate,

            indexingRate,

            processingLatency,

            searchLatency,

            storage,

            categories,
        };
    }

    /* ----------------------------------------------------------------------
       Normalize health
       ---------------------------------------------------------------------- */

    function normalizeHealth(
        raw
    ) {
        const source =
            unwrapResponse(
                raw
            );

        const overall =
            extractValue(
                source,
                [
                    "status",
                    "health",
                    "overall_status",
                ]
            );

        const services =
            isObject(
                extractValue(
                    source,
                    [
                        "services",
                        "checks",
                        "components",
                    ]
                )
            )
                ? extractValue(
                    source,
                    [
                        "services",
                        "checks",
                        "components",
                    ]
                )
                : {};

        const vector =
            extractValue(
                source,
                [
                    "vector_store",
                    "vector",
                    "services.vector_store",
                    "services.vector",
                    "checks.vector_store",
                ]
            );

        const ocr =
            extractValue(
                source,
                [
                    "ocr",
                    "services.ocr",
                    "checks.ocr",
                ]
            );

        const gemini =
            extractValue(
                source,
                [
                    "gemini",
                    "ai",
                    "ai_service",
                    "services.gemini",
                    "services.ai",
                    "checks.gemini",
                ]
            );

        return {
            raw: source,

            status:
                normalizeStatus(
                    overall
                ),

            services,

            vector:
                normalizeStatus(
                    vector ??
                    services.vector_store ??
                    services.vector
                ),

            ocr:
                normalizeStatus(
                    ocr ??
                    services.ocr
                ),

            gemini:
                normalizeStatus(
                    gemini ??
                    services.gemini ??
                    services.ai
                ),
        };
    }

    /* ----------------------------------------------------------------------
       Normalize readiness
       ---------------------------------------------------------------------- */

    function normalizeReadiness(
        raw
    ) {
        const source =
            unwrapResponse(
                raw
            );

        const value =
            extractValue(
                source,
                [
                    "ready",
                    "status",
                    "healthy",
                ]
            );

        let ready = false;

        if (
            typeof value ===
            "boolean"
        ) {
            ready = value;
        } else {
            ready =
                normalizeStatus(
                    value
                ) === "healthy";
        }

        return {
            raw: source,
            ready,
            status:
                ready
                    ? "healthy"
                    : "unavailable",
        };
    }

    /* ----------------------------------------------------------------------
       Category metrics
       ---------------------------------------------------------------------- */

    function renderCategories(
        categories
    ) {
        const container =
            queryFirst(
                CONFIG.selectors.categoryMetrics
            );

        if (!container) {
            return;
        }

        /*
         * Backend did not provide categories.
         * Do not invent them.
         */

        if (
            categories === null ||
            categories ===
                undefined
        ) {
            container.textContent =
                CONFIG.fallbackText;

            return;
        }

        const entries = [];

        if (isArray(categories)) {
            categories.forEach(
                (item) => {
                    if (
                        isObject(
                            item
                        )
                    ) {
                        const name =
                            firstDefined(
                                item.category,
                                item.name,
                                item.label
                            );

                        const count =
                            toNumber(
                                firstDefined(
                                    item.count,
                                    item.total,
                                    item.value
                                )
                            );

                        if (
                            name &&
                            count !==
                                null
                        ) {
                            entries.push({
                                name:
                                    String(
                                        name
                                    ),
                                count,
                            });
                        }
                    }
                }
            );
        } else if (
            isObject(categories)
        ) {
            Object.entries(
                categories
            ).forEach(
                ([name, value]) => {
                    const count =
                        toNumber(
                            value
                        );

                    if (
                        count !==
                        null
                    ) {
                        entries.push({
                            name,
                            count,
                        });
                    }
                }
            );
        }

        if (!entries.length) {
            container.textContent =
                CONFIG.fallbackText;

            return;
        }

        entries.sort(
            (a, b) =>
                b.count -
                a.count
        );

        const fragment =
            document.createDocumentFragment();

        const total =
            entries.reduce(
                (sum, item) =>
                    sum +
                    item.count,
                0
            );

        entries
            .slice(0, 8)
            .forEach(
                (item) => {
                    const row =
                        document.createElement(
                            "div"
                        );

                    row.className =
                        "stats-category-row";

                    const header =
                        document.createElement(
                            "div"
                        );

                    header.className =
                        "stats-category-row__header";

                    const name =
                        document.createElement(
                            "span"
                        );

                    name.className =
                        "stats-category-row__name";

                    name.textContent =
                        item.name;

                    const count =
                        document.createElement(
                            "span"
                        );

                    count.className =
                        "stats-category-row__count";

                    count.textContent =
                        formatNumber(
                            item.count
                        );

                    header.append(
                        name,
                        count
                    );

                    const track =
                        document.createElement(
                            "div"
                        );

                    track.className =
                        "stats-category-row__track";

                    const fill =
                        document.createElement(
                            "div"
                        );

                    fill.className =
                        "stats-category-row__fill";

                    const ratio =
                        total > 0
                            ? clamp(
                                item.count /
                                    total,
                                0,
                                1
                            )
                            : 0;

                    fill.style.setProperty(
                        "--category-progress",
                        `${Math.round(
                            ratio * 100
                        )}%`
                    );

                    track.appendChild(
                        fill
                    );

                    row.append(
                        header,
                        track
                    );

                    fragment.appendChild(
                        row
                    );
                }
            );

        while (
            container.firstChild
        ) {
            container.removeChild(
                container.firstChild
            );
        }

        container.appendChild(
            fragment
        );
    }

    /* ----------------------------------------------------------------------
       Render primary metrics
       ---------------------------------------------------------------------- */

    function renderPrimaryStats(
        stats
    ) {
        setText(
            CONFIG.selectors.memories,
            stats.memories !==
                null
                ? formatNumber(
                    stats.memories
                )
                : CONFIG.fallbackText
        );

        setText(
            CONFIG.selectors.searches,
            stats.searches !==
                null
                ? formatNumber(
                    stats.searches
                )
                : CONFIG.fallbackText
        );

        setText(
            CONFIG.selectors.successRate,
            stats.successRate !==
                null
                ? formatPercentage(
                    stats.successRate
                )
                : CONFIG.fallbackText
        );

        setText(
            CONFIG.selectors.indexingRate,
            stats.indexingRate !==
                null
                ? formatPercentage(
                    stats.indexingRate
                )
                : CONFIG.fallbackText
        );
    }

    /* ----------------------------------------------------------------------
       Render performance metrics
       ---------------------------------------------------------------------- */

    function renderPerformance(
        stats
    ) {
        setText(
            CONFIG.selectors.processingLatency,
            stats.processingLatency !==
                null
                ? formatLatency(
                    stats.processingLatency
                )
                : CONFIG.fallbackText
        );

        setText(
            CONFIG.selectors.searchLatency,
            stats.searchLatency !==
                null
                ? formatLatency(
                    stats.searchLatency
                )
                : CONFIG.fallbackText
        );

        let storageValue =
            CONFIG.fallbackText;

        if (
            isObject(
                stats.storage
            )
        ) {
            const bytes =
                firstDefined(
                    stats.storage.bytes,
                    stats.storage.size,
                    stats.storage.used
                );

            if (
                bytes !==
                null
            ) {
                storageValue =
                    formatBytes(
                        bytes
                    );
            }
        } else if (
            stats.storage !==
                null
        ) {
            storageValue =
                formatBytes(
                    stats.storage
                );
        }

        setText(
            CONFIG.selectors.storage,
            storageValue
        );
    }

    /* ----------------------------------------------------------------------
       Render service status
       ---------------------------------------------------------------------- */

    function renderService(
        selectors,
        status
    ) {
        const normalized =
            normalizeStatus(
                status
            );

        setText(
            selectors,
            statusLabel(
                normalized
            )
        );

        setDataStatus(
            selectors,
            normalized
        );
    }

    function renderHealth(
        health,
        readiness
    ) {
        renderService(
            CONFIG.selectors.vectorHealth,
            health.vector
        );

        renderService(
            CONFIG.selectors.ocrHealth,
            health.ocr
        );

        renderService(
            CONFIG.selectors.geminiStatus,
            health.gemini
        );

        renderService(
            CONFIG.selectors.healthStatus,
            health.status
        );

        renderService(
            CONFIG.selectors.readyStatus,
            readiness.status
        );
    }

    /* ----------------------------------------------------------------------
       Last updated
       ---------------------------------------------------------------------- */

    function renderLastUpdated() {
        const element =
            queryFirst(
                CONFIG.selectors.lastUpdated
            );

        if (!element) {
            return;
        }

        if (!state.lastUpdated) {
            element.textContent =
                CONFIG.fallbackText;

            return;
        }

        try {
            element.textContent =
                `Updated ${new Intl.DateTimeFormat(
                    undefined,
                    {
                        hour:
                            "2-digit",
                        minute:
                            "2-digit",
                        second:
                            "2-digit",
                    }
                ).format(
                    state.lastUpdated
                )}`;
        } catch {
            element.textContent =
                "Updated just now";
        }
    }

    /* ----------------------------------------------------------------------
       Loading state
       ---------------------------------------------------------------------- */

    function renderLoading() {
        state.loading = true;

        setAttribute(
            CONFIG.selectors.memories,
            "aria-busy",
            "true"
        );

        setAttribute(
            CONFIG.selectors.successRate,
            "aria-busy",
            "true"
        );

        setAttribute(
            CONFIG.selectors.searches,
            "aria-busy",
            "true"
        );

        setAttribute(
            CONFIG.selectors.indexingRate,
            "aria-busy",
            "true"
        );

        dispatch(
            CONFIG.events.statsLoading,
            {
                timestamp:
                    Date.now(),
            }
        );
    }

    /* ----------------------------------------------------------------------
       Render error
       ---------------------------------------------------------------------- */

    function renderError(
        error
    ) {
        state.loading = false;

        state.error =
            error instanceof Error
                ? error.message
                : toSafeString(
                    error,
                    "Statistics unavailable"
                );

        /*
         * Do not destroy existing valid metrics.
         * Only mark the dashboard as degraded.
         */

        const health =
            queryFirst(
                CONFIG.selectors.healthStatus
            );

        if (health) {
            health.textContent =
                "Unavailable";

            health.dataset.status =
                "unavailable";
        }

        dispatch(
            CONFIG.events.statsError,
            {
                error:
                    state.error,
                timestamp:
                    Date.now(),
            }
        );
    }

    /* ----------------------------------------------------------------------
       Render complete state
       ---------------------------------------------------------------------- */

    function renderComplete() {
        state.loading = false;
        state.error = null;
        state.lastUpdated =
            new Date();

        setAttribute(
            CONFIG.selectors.memories,
            "aria-busy",
            "false"
        );

        setAttribute(
            CONFIG.selectors.successRate,
            "aria-busy",
            "false"
        );

        setAttribute(
            CONFIG.selectors.searches,
            "aria-busy",
            "false"
        );

        setAttribute(
            CONFIG.selectors.indexingRate,
            "aria-busy",
            "false"
        );

        renderLastUpdated();
    }

    /* ----------------------------------------------------------------------
       Main refresh
       ---------------------------------------------------------------------- */

    async function refresh(
        options = {}
    ) {
        const requestId =
            ++state.requestId;

        /*
         * Cancel previous refresh.
         */

        if (
            state.requestController
        ) {
            try {
                state.requestController.abort();
            } catch {
                // Ignore.
            }
        }

        state.requestController =
            new AbortController();

        renderLoading();

        const startedAt =
            performance.now();

        try {
            const [
                statsResponse,
                healthResponse,
                readyResponse,
            ] =
                await Promise.allSettled(
                    [
                        request(
                            CONFIG.endpoints.stats
                        ),

                        request(
                            CONFIG.endpoints.health
                        ),

                        request(
                            CONFIG.endpoints.ready
                        ),
                    ]
                );

            /*
             * Ignore stale response.
             */

            if (
                requestId !==
                state.requestId
            ) {
                return null;
            }

            /*
             * Stats
             */

            if (
                statsResponse.status ===
                "fulfilled"
            ) {
                state.stats =
                    normalizeStats(
                        statsResponse.value
                    );

                renderPrimaryStats(
                    state.stats
                );

                renderPerformance(
                    state.stats
                );

                renderCategories(
                    state.stats.categories
                );
            }

            /*
             * Health
             */

            if (
                healthResponse.status ===
                "fulfilled"
            ) {
                state.health =
                    normalizeHealth(
                        healthResponse.value
                    );
            } else {
                state.health =
                    normalizeHealth(
                        null
                    );
            }

            /*
             * Readiness
             */

            if (
                readyResponse.status ===
                "fulfilled"
            ) {
                state.readiness =
                    normalizeReadiness(
                        readyResponse.value
                    );
            } else {
                state.readiness =
                    normalizeReadiness(
                        null
                    );
            }

            renderHealth(
                state.health,
                state.readiness
            );

            /*
             * If all requests failed,
             * surface an actual error.
             */

            const allFailed =
                statsResponse.status ===
                    "rejected" &&
                healthResponse.status ===
                    "rejected" &&
                readyResponse.status ===
                    "rejected";

            if (allFailed) {
                throw (
                    statsResponse.reason ||
                    healthResponse.reason ||
                    readyResponse.reason ||
                    new Error(
                        "Unable to retrieve system statistics."
                    )
                );
            }

            renderComplete();

            const latency =
                Math.round(
                    performance.now() -
                    startedAt
                );

            dispatch(
                CONFIG.events.statsComplete,
                {
                    stats:
                        state.stats,
                    health:
                        state.health,
                    readiness:
                        state.readiness,
                    latency,
                    refreshed:
                        Boolean(
                            options.manual
                        ),
                    timestamp:
                        Date.now(),
                }
            );

            return {
                stats:
                    state.stats,
                health:
                    state.health,
                readiness:
                    state.readiness,
            };
        } catch (error) {
            if (
                requestId !==
                state.requestId
            ) {
                return null;
            }

            renderError(
                error
            );

            return null;
        }
    }

    /* ----------------------------------------------------------------------
       Refresh interval
       ---------------------------------------------------------------------- */

    function startAutoRefresh() {
        stopAutoRefresh();

        state.refreshTimer =
            window.setInterval(
                () => {
                    /*
                     * Avoid background traffic when
                     * the browser tab is hidden.
                     */

                    if (
                        document.hidden
                    ) {
                        return;
                    }

                    refresh({
                        manual: false,
                    });
                },
                CONFIG.refreshInterval
            );
    }

    function stopAutoRefresh() {
        if (
            state.refreshTimer
        ) {
            window.clearInterval(
                state.refreshTimer
            );

            state.refreshTimer =
                null;
        }
    }

    /* ----------------------------------------------------------------------
       Refresh button integration
       ---------------------------------------------------------------------- */

    function bindRefreshButtons() {
        const buttons =
            queryAll(
                CONFIG.selectors.refreshButtons
            );

        buttons.forEach(
            (button) => {
                if (
                    button.dataset
                        .statsBound ===
                    "true"
                ) {
                    return;
                }

                button.dataset
                    .statsBound =
                    "true";

                button.addEventListener(
                    "click",
                    async (
                        event
                    ) => {
                        event.preventDefault();

                        button.setAttribute(
                            "aria-busy",
                            "true"
                        );

                        button.classList.add(
                            "is-refreshing"
                        );

                        dispatch(
                            CONFIG.events.statsRefresh,
                            {
                                source:
                                    "button",
                            }
                        );

                        try {
                            await refresh({
                                manual:
                                    true,
                            });
                        } finally {
                            button.removeAttribute(
                                "aria-busy"
                            );

                            button.classList.remove(
                                "is-refreshing"
                            );
                        }
                    }
                );
            }
        );
    }

    /* ----------------------------------------------------------------------
       Visibility handling
       ---------------------------------------------------------------------- */

    function handleVisibilityChange() {
        if (
            document.hidden
        ) {
            return;
        }

        /*
         * Refresh once when returning to
         * the dashboard if the existing data
         * is stale.
         */

        if (
            state.lastUpdated
        ) {
            const age =
                Date.now() -
                state.lastUpdated.getTime();

            if (
                age >
                CONFIG.refreshInterval
            ) {
                refresh({
                    manual: false,
                });
            }
        }
    }

    /* ----------------------------------------------------------------------
       External refresh event
       ---------------------------------------------------------------------- */

    function handleRefreshEvent() {
        refresh({
            manual: false,
        });
    }

    /* ----------------------------------------------------------------------
       Public API
       ---------------------------------------------------------------------- */

    const StatsController = {
        async init(
            options = {}
        ) {
            if (
                state.initialized
            ) {
                return this;
            }

            state.initialized =
                true;

            bindRefreshButtons();

            window.addEventListener(
                CONFIG.events.refresh,
                handleRefreshEvent
            );

            document.addEventListener(
                "visibilitychange",
                handleVisibilityChange
            );

            if (
                options.autoRefresh !==
                false
            ) {
                startAutoRefresh();
            }

            await refresh({
                manual: false,
            });

            return this;
        },

        refresh(
            options = {}
        ) {
            return refresh(
                options
            );
        },

        getState() {
            return {
                initialized:
                    state.initialized,

                loading:
                    state.loading,

                error:
                    state.error,

                stats:
                    state.stats,

                health:
                    state.health,

                readiness:
                    state.readiness,

                lastUpdated:
                    state.lastUpdated,
            };
        },

        getStats() {
            return state.stats;
        },

        getHealth() {
            return state.health;
        },

        getReadiness() {
            return state.readiness;
        },

        startAutoRefresh() {
            startAutoRefresh();

            return this;
        },

        stopAutoRefresh() {
            stopAutoRefresh();

            return this;
        },

        destroy() {
            stopAutoRefresh();

            if (
                state.requestController
            ) {
                try {
                    state.requestController.abort();
                } catch {
                    // Ignore.
                }
            }

            window.removeEventListener(
                CONFIG.events.refresh,
                handleRefreshEvent
            );

            document.removeEventListener(
                "visibilitychange",
                handleVisibilityChange
            );

            state.initialized =
                false;

            return this;
        },
    };

    /* ----------------------------------------------------------------------
       Expose
       ---------------------------------------------------------------------- */

    window.MemoryOS.stats =
        StatsController;

    /* ----------------------------------------------------------------------
       Initialize after DOM
       ---------------------------------------------------------------------- */

    function boot() {
        StatsController.init();
    }

    if (
        document.readyState ===
        "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            boot,
            {
                once: true,
            }
        );
    } else {
        boot();
    }

    /* ----------------------------------------------------------------------
       Development diagnostics
       ---------------------------------------------------------------------- */

    if (
        window.location?.hostname ===
            "localhost" ||
        window.location?.hostname ===
            "127.0.0.1"
    ) {
        console.info(
            "%cMemoryOS Stats%c initialized",
            "font-weight:700;",
            "font-weight:400;"
        );
    }
})();