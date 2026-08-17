/* ==========================================================================
   MemoryOS — Results Controller
   frontend/js/results.js

   PREMIUM VERSION

   Responsibilities:
   - Receive search results from SearchController
   - Render premium memory cards
   - Safely render API-controlled data
   - Handle confidence / scores
   - Render semantic match explanations
   - Handle loading / empty / error / degraded states
   - Load screenshot previews through FastAPI
   - Convert Windows filesystem paths into API image URLs
   - Handle image loading and fallbacks
   - Open memory detail drawer through events
   - Support keyboard accessibility
   - Remain resilient to backend response variations
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
        selectors: {
            grid: [
                "#memoryGrid",
                "#resultsGrid",
                "[data-memory-grid]",
                "[data-results-grid]",
            ],

            count: [
                "#resultCount",
                "#resultsCount",
                "[data-result-count]",
            ],

            status: [
                "#resultsStatus",
                "[data-results-status]",
            ],

            empty: [
                "#emptyResults",
                "[data-empty-results]",
            ],

            loading: [
                "#resultsLoading",
                "[data-results-loading]",
            ],

            error: [
                "#resultsError",
                "[data-results-error]",
            ],
        },

        events: Object.freeze({
            searchComplete: "memoryos:search-complete",
            searchLoading: "memoryos:search-loading",
            searchEmpty: "memoryos:search-empty",
            searchError: "memoryos:search-error",
            searchCleared: "memoryos:search-cleared",
            openMemory: "memoryos:open-memory",
            refresh: "memoryos:refresh",
            resultsRendered: "memoryos:results-rendered",
        }),

        defaults: Object.freeze({
            confidence: 0,
            category: "Memory",
            intent: "General",
            imageAlt: "Memory screenshot",
            placeholder: "No preview available",
        }),

        limits: Object.freeze({
            title: 140,
            summary: 280,
            entity: 80,
            metadata: 120,
            explanation: 160,
            entities: 5,
        }),
    });

    /* ----------------------------------------------------------------------
       State
       ---------------------------------------------------------------------- */

    const state = {
        initialized: false,
        loading: false,
        error: null,
        query: "",
        results: [],
        total: 0,
        renderedAt: null,
        container: null,
        countElement: null,
        statusElement: null,
    };

    /* ----------------------------------------------------------------------
       Utilities
       ---------------------------------------------------------------------- */

    function isObject(value) {
        return value !== null && typeof value === "object";
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

    function toSafeString(value, fallback = "") {
        if (value === undefined || value === null) {
            return fallback;
        }

        if (typeof value === "string") {
            return value.trim();
        }

        if (
            typeof value === "number" ||
            typeof value === "boolean"
        ) {
            return String(value);
        }

        return fallback;
    }

    function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }

    function normalizeScore(value) {
        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            return 0;
        }

        let number = Number(value);

        if (!Number.isFinite(number)) {
            return 0;
        }

        /*
         * Backend may return:
         *
         * 0.93
         * 93
         * "0.93"
         */

        if (number > 1 && number <= 100) {
            number /= 100;
        }

        return clamp(number, 0, 1);
    }

    function percentage(value) {
        return Math.round(normalizeScore(value) * 100);
    }

    function truncate(value, limit) {
        const text = toSafeString(value);

        if (text.length <= limit) {
            return text;
        }

        return `${text.slice(0, limit - 1).trim()}…`;
    }

    function normalizeTextArray(value) {
        if (isArray(value)) {
            return value
                .map((item) => {
                    if (typeof item === "string") {
                        return item.trim();
                    }

                    if (isObject(item)) {
                        return toSafeString(
                            firstDefined(
                                item.name,
                                item.value,
                                item.text,
                                item.label
                            )
                        );
                    }

                    return "";
                })
                .filter(Boolean);
        }

        if (typeof value === "string") {
            return value
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean);
        }

        return [];
    }

    function normalizeCategory(value) {
        const category = toSafeString(
            value,
            CONFIG.defaults.category
        );

        if (!category) {
            return CONFIG.defaults.category;
        }

        return (
            category.charAt(0).toUpperCase() +
            category.slice(1)
        );
    }

    function normalizeIntent(value) {
        const intent = toSafeString(
            value,
            CONFIG.defaults.intent
        );

        if (!intent) {
            return CONFIG.defaults.intent;
        }

        return (
            intent.charAt(0).toUpperCase() +
            intent.slice(1)
        );
    }

    /* ----------------------------------------------------------------------
       IMPORTANT — IMAGE URL RESOLVER
       ----------------------------------------------------------------------

       Backend currently returns:

       C:\Users\GODWIN\Pictures\Screenshots\Screenshot (17).png

       Browsers cannot use that path directly.

       FastAPI exposes:

       GET /upload/file/{memory_id}

       Therefore:

       memory.image
          +
       memory.id

       becomes:

       http://127.0.0.1:8000/upload/file/{memory_id}

       This is the main screenshot-preview fix.
       ---------------------------------------------------------------------- */

    function resolveApiBase() {
        /*
         * First preference:
         * MemoryOS.config.apiBase
         */

        if (
            window.MemoryOS &&
            window.MemoryOS.config
        ) {
            const config = window.MemoryOS.config;

            const candidate = firstDefined(
                config.apiBase,
                config.API_BASE,
                config.apiUrl,
                config.API_URL,
                config.backendUrl,
                config.BACKEND_URL,
                config.baseUrl,
                config.BASE_URL
            );

            if (candidate) {
                return String(candidate).replace(/\/+$/, "");
            }
        }

        /*
         * If frontend is running on Vite:
         *
         * http://localhost:5173
         *
         * backend:
         *
         * http://127.0.0.1:8000
         */

        return "http://127.0.0.1:8000";
    }

    function resolveImageUrl(rawImage, memoryId) {
        const image = toSafeString(rawImage);
        const id = toSafeString(memoryId);

        /*
         * No image and no memory ID.
         */

        if (!image && !id) {
            return "";
        }

        const apiBase = resolveApiBase();

        /*
         * BEST CASE:
         * Memory ID exists.
         *
         * Always prefer the backend file endpoint.
         *
         * This fixes Windows paths such as:
         *
         * C:\Users\GODWIN\Pictures\Screenshots\...
         */

        if (id) {
            return `${apiBase}/upload/file/${encodeURIComponent(id)}`;
        }

        /*
         * Already a full HTTP URL.
         */

        if (
            image.startsWith("http://") ||
            image.startsWith("https://") ||
            image.startsWith("blob:") ||
            image.startsWith("data:")
        ) {
            return image;
        }

        /*
         * Backend relative path.
         */

        if (image.startsWith("/")) {
            return `${apiBase}${image}`;
        }

        /*
         * Generic relative path.
         */

        return `${apiBase}/${image.replace(/^\/+/, "")}`;
    }

    /* ----------------------------------------------------------------------
       Normalize memory
       ---------------------------------------------------------------------- */

    function normalizeMemory(raw) {
        if (!isObject(raw)) {
            return null;
        }

        const nestedMemory = isObject(raw.memory)
            ? raw.memory
            : {};

        const metadata = isObject(raw.metadata)
            ? raw.metadata
            : isObject(nestedMemory.metadata)
                ? nestedMemory.metadata
                : {};

        const ai = isObject(raw.ai_analysis)
            ? raw.ai_analysis
            : isObject(raw.analysis)
                ? raw.analysis
                : {};

        const explanation = isObject(raw.explanation)
            ? raw.explanation
            : isObject(raw.match_explanation)
                ? raw.match_explanation
                : {};

        const entities = normalizeTextArray(
            firstDefined(
                raw.entities,
                ai.entities,
                metadata.entities
            )
        );

        const confidence = firstDefined(
            raw.confidence,
            raw.confidence_score,
            raw.match_confidence,
            raw.score,
            raw.similarity,
            raw.similarity_score
        );

        const rawImage = firstDefined(
            raw.original_image_url,
            raw.originalImage,
            raw.image_url,
            raw.image,
            raw.preview_url,
            raw.thumbnail_url,
            raw.thumbnail,
            raw.file_url,
            raw.path,
            nestedMemory.original_image_url,
            nestedMemory.image_url,
            nestedMemory.preview_url,
            nestedMemory.thumbnail_url,
            metadata.original_image_url,
            metadata.image_url,
            metadata.preview_url,
            metadata.thumbnail_url,
            metadata.path
        );

        const title = firstDefined(
            raw.title,
            raw.name,
            ai.title,
            metadata.title,
            raw.filename,
            "Untitled memory"
        );

        const summary = firstDefined(
            raw.summary,
            raw.description,
            ai.summary,
            raw.semantic_summary,
            metadata.summary,
            raw.semantic_text
        );

        const category = firstDefined(
            raw.category,
            ai.category,
            metadata.category
        );

        const intent = firstDefined(
            raw.intent,
            ai.intent,
            metadata.intent
        );

        const memoryId = firstDefined(
            raw.memory_id,
            raw.id,
            raw.uuid,
            metadata.memory_id,
            metadata.id
        );

        const whyMatched = normalizeExplanation(
            raw,
            explanation
        );

        const safeId = toSafeString(memoryId);

        return {
            raw,

            id: safeId,

            title: truncate(
                title,
                CONFIG.limits.title
            ),

            summary: truncate(
                summary ||
                "AI-generated memory representation",
                CONFIG.limits.summary
            ),

            category: normalizeCategory(category),

            intent: normalizeIntent(intent),

            entities: entities
                .slice(0, CONFIG.limits.entities)
                .map((entity) =>
                    truncate(
                        entity,
                        CONFIG.limits.entity
                    )
                ),

            /*
             * IMPORTANT:
             *
             * Do NOT use Windows filesystem path as
             * the final <img src>.
             *
             * Convert it to FastAPI endpoint.
             */

            image: resolveImageUrl(
                rawImage,
                safeId
            ),

            originalImage: toSafeString(rawImage),

            confidence: normalizeScore(
                confidence
            ),

            score: normalizeScore(
                firstDefined(
                    raw.score,
                    raw.rerank_score,
                    raw.similarity,
                    raw.similarity_score,
                    confidence
                )
            ),

            ocrText: toSafeString(
                firstDefined(
                    raw.ocr_text,
                    raw.ocr,
                    raw.text,
                    metadata.ocr_text
                )
            ),

            semanticText: toSafeString(
                firstDefined(
                    raw.semantic_text,
                    raw.semantic_representation,
                    ai.semantic_text,
                    metadata.semantic_text
                )
            ),

            createdAt: toSafeString(
                firstDefined(
                    raw.created_at,
                    raw.createdAt,
                    raw.timestamp,
                    metadata.created_at
                )
            ),

            status: toSafeString(
                firstDefined(
                    raw.status,
                    raw.processing_status,
                    metadata.status
                ),
                "ready"
            ),

            whyMatched,
        };
    }

    /* ----------------------------------------------------------------------
       Explanation
       ---------------------------------------------------------------------- */

    function normalizeExplanation(
        raw,
        explanation
    ) {
        const items = [];

        const possibleLists = [
            raw.why_matched,
            raw.match_reasons,
            raw.reasons,
            explanation.reasons,
            explanation.items,
            explanation.matches,
        ];

        for (const source of possibleLists) {
            if (!isArray(source)) {
                continue;
            }

            for (const item of source) {
                if (typeof item === "string") {
                    items.push({
                        label: truncate(
                            item,
                            CONFIG.limits.explanation
                        ),
                        strength: "positive",
                    });

                    continue;
                }

                if (isObject(item)) {
                    const label = firstDefined(
                        item.label,
                        item.reason,
                        item.message,
                        item.text,
                        item.name
                    );

                    if (label) {
                        items.push({
                            label: truncate(
                                label,
                                CONFIG.limits.explanation
                            ),
                            strength: toSafeString(
                                firstDefined(
                                    item.strength,
                                    item.type,
                                    item.status
                                ),
                                "positive"
                            ),
                        });
                    }
                }
            }
        }

        const structuredCandidates = [
            [
                "Strong semantic similarity",
                firstDefined(
                    raw.semantic_similarity,
                    raw.semantic_score,
                    explanation.semantic_similarity
                ),
            ],
            [
                "Keyword overlap",
                firstDefined(
                    raw.keyword_overlap,
                    raw.keyword_score,
                    explanation.keyword_overlap
                ),
            ],
            [
                "Entity match",
                firstDefined(
                    raw.entity_match,
                    raw.entity_score,
                    explanation.entity_match
                ),
            ],
            [
                "Category match",
                firstDefined(
                    raw.category_match,
                    raw.category_score,
                    explanation.category_match
                ),
            ],
            [
                "Intent match",
                firstDefined(
                    raw.intent_match,
                    raw.intent_score,
                    explanation.intent_match
                ),
            ],
        ];

        for (const [
            label,
            value,
        ] of structuredCandidates) {
            if (
                value === undefined ||
                value === null
            ) {
                continue;
            }

            const score = normalizeScore(value);

            if (score >= 0.5) {
                items.push({
                    label,
                    strength:
                        score >= 0.75
                            ? "strong"
                            : "positive",
                });
            }
        }

        const seen = new Set();

        return items
            .filter((item) => {
                const key =
                    item.label.toLowerCase();

                if (seen.has(key)) {
                    return false;
                }

                seen.add(key);

                return true;
            })
            .slice(0, 5);
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
                // Ignore malformed selectors.
            }
        }

        return null;
    }

    function createElement(
        tag,
        className = "",
        text = ""
    ) {
        const element =
            document.createElement(tag);

        if (className) {
            element.className =
                className;
        }

        if (text !== "") {
            element.textContent =
                text;
        }

        return element;
    }

    function setText(
        element,
        value,
        fallback = ""
    ) {
        if (!element) {
            return;
        }

        element.textContent =
            toSafeString(
                value,
                fallback
            );
    }

    function clearElement(element) {
        if (!element) {
            return;
        }

        while (element.firstChild) {
            element.removeChild(
                element.firstChild
            );
        }
    }

    function createIcon(
        name,
        label = ""
    ) {
        const icon =
            createElement("i");

        icon.setAttribute(
            "data-lucide",
            name
        );

        if (label) {
            icon.setAttribute(
                "aria-label",
                label
            );
        }

        icon.setAttribute(
            "aria-hidden",
            "true"
        );

        return icon;
    }

    function refreshIcons(
        root = document
    ) {
        try {
            if (
                window.lucide &&
                typeof window.lucide
                    .createIcons ===
                    "function"
            ) {
                window.lucide.createIcons({
                    root,
                });
            }
        } catch {
            // Decorative only.
        }
    }

    /* ----------------------------------------------------------------------
       Container
       ---------------------------------------------------------------------- */

    function resolveElements() {
        state.container =
            queryFirst(
                CONFIG.selectors.grid
            );

        state.countElement =
            queryFirst(
                CONFIG.selectors.count
            );

        state.statusElement =
            queryFirst(
                CONFIG.selectors.status
            );

        return Boolean(
            state.container
        );
    }

    /* ----------------------------------------------------------------------
       Count
       ---------------------------------------------------------------------- */

    function renderCount(count) {
        const safeCount =
            Number.isFinite(
                Number(count)
            )
                ? Math.max(
                    0,
                    Number(count)
                )
                : 0;

        if (state.countElement) {
            state.countElement.textContent =
                `${safeCount} ${
                    safeCount === 1
                        ? "memory"
                        : "memories"
                }`;
        }

        if (state.statusElement) {
            state.statusElement.textContent =
                safeCount === 0
                    ? "No matching memories"
                    : `${safeCount} matching ${
                        safeCount === 1
                            ? "memory"
                            : "memories"
                    }`;
        }
    }

    /* ----------------------------------------------------------------------
       Visibility
       ---------------------------------------------------------------------- */

    function setVisibility(
        selectorList,
        visible
    ) {
        for (const selector of selectorList) {
            let elements = [];

            try {
                elements =
                    document.querySelectorAll(
                        selector
                    );
            } catch {
                continue;
            }

            elements.forEach(
                (element) => {
                    element.hidden =
                        !visible;

                    element.setAttribute(
                        "aria-hidden",
                        visible
                            ? "false"
                            : "true"
                    );
                }
            );
        }
    }

    function setLoading(
        visible
    ) {
        state.loading =
            visible;

        setVisibility(
            CONFIG.selectors.loading,
            visible
        );
    }

    function setEmpty(
        visible
    ) {
        setVisibility(
            CONFIG.selectors.empty,
            visible
        );
    }

    function setError(
        visible
    ) {
        setVisibility(
            CONFIG.selectors.error,
            visible
        );
    }

    /* ----------------------------------------------------------------------
       Premium skeletons
       ---------------------------------------------------------------------- */

    function renderSkeletons(
        count = 6
    ) {
        if (!state.container) {
            return;
        }

        clearElement(
            state.container
        );

        const fragment =
            document.createDocumentFragment();

        const safeCount =
            clamp(
                Number(count) || 6,
                1,
                12
            );

        for (
            let index = 0;
            index < safeCount;
            index += 1
        ) {
            const card =
                createElement(
                    "article",
                    "memory-card memory-card--skeleton"
                );

            card.setAttribute(
                "aria-hidden",
                "true"
            );

            const media =
                createElement(
                    "div",
                    "memory-card__media skeleton"
                );

            const body =
                createElement(
                    "div",
                    "memory-card__body"
                );

            const title =
                createElement(
                    "div",
                    "skeleton skeleton--title"
                );

            const line =
                createElement(
                    "div",
                    "skeleton skeleton--line"
                );

            const lineShort =
                createElement(
                    "div",
                    "skeleton skeleton--line skeleton--short"
                );

            const footer =
                createElement(
                    "div",
                    "memory-card__footer"
                );

            const chip =
                createElement(
                    "div",
                    "skeleton skeleton--chip"
                );

            const score =
                createElement(
                    "div",
                    "skeleton skeleton--score"
                );

            body.append(
                title,
                line,
                lineShort,
                footer
            );

            footer.append(
                chip,
                score
            );

            card.append(
                media,
                body
            );

            fragment.appendChild(
                card
            );
        }

        state.container.appendChild(
            fragment
        );
    }

    /* ----------------------------------------------------------------------
       Image
       ---------------------------------------------------------------------- */

    function createMemoryImage(
        memory
    ) {
        const wrapper =
            createElement(
                "div",
                "memory-card__image-wrap"
            );

        const media =
            createElement(
                "div",
                "memory-card__media"
            );

        const fallback =
            createElement(
                "div",
                "memory-card__image-fallback"
            );

        fallback.appendChild(
            createIcon(
                "image-off"
            )
        );

        fallback.appendChild(
            createElement(
                "span",
                "",
                CONFIG.defaults.placeholder
            )
        );

        const image =
            document.createElement(
                "img"
            );

        image.className =
            "memory-card__image";

        image.loading =
            "lazy";

        image.decoding =
            "async";

        image.alt =
            truncate(
                memory.title ||
                CONFIG.defaults.imageAlt,
                120
            );

        /*
         * Start with image hidden.
         *
         * It becomes visible only after
         * the browser successfully loads it.
         */

        image.hidden = true;

        fallback.hidden = true;

        /*
         * Loading state.
         */

        media.classList.add(
            "memory-card__media--loading"
        );

        /*
         * SUCCESS
         */

        image.addEventListener(
            "load",
            () => {
                image.hidden = false;

                fallback.hidden = true;

                media.classList.remove(
                    "memory-card__media--loading"
                );

                media.classList.remove(
                    "memory-card__media--fallback"
                );

                image.classList.add(
                    "memory-card__image--loaded"
                );
            },
            {
                once: true,
            }
        );

        /*
         * FAILURE
         */

        image.addEventListener(
            "error",
            () => {
                image.hidden = true;

                fallback.hidden = false;

                media.classList.remove(
                    "memory-card__media--loading"
                );

                media.classList.add(
                    "memory-card__media--fallback"
                );

                console.warn(
                    "[MemoryOS] Screenshot preview failed:",
                    {
                        memoryId:
                            memory.id,
                        imageUrl:
                            memory.image,
                        originalPath:
                            memory.originalImage,
                    }
                );
            },
            {
                once: true,
            }
        );

        /*
         * IMPORTANT:
         *
         * memory.image is now:
         *
         * http://127.0.0.1:8000/upload/file/mem_xxx
         *
         * instead of:
         *
         * C:\Users\...\Screenshot.png
         */

        if (memory.image) {
            image.src =
                memory.image;
        } else {
            fallback.hidden =
                false;

            media.classList.remove(
                "memory-card__media--loading"
            );

            media.classList.add(
                "memory-card__media--fallback"
            );
        }

        media.append(
            image,
            fallback
        );

        wrapper.appendChild(
            media
        );

        return wrapper;
    }

    /* ----------------------------------------------------------------------
       Chip
       ---------------------------------------------------------------------- */

    function createChip(
        iconName,
        text,
        modifier = ""
    ) {
        const chip =
            createElement(
                "span",
                `memory-chip ${modifier}`.trim()
            );

        chip.appendChild(
            createIcon(iconName)
        );

        chip.appendChild(
            createElement(
                "span",
                "memory-chip__label",
                truncate(
                    text,
                    CONFIG.limits.metadata
                )
            )
        );

        return chip;
    }

    /* ----------------------------------------------------------------------
       Confidence
       ---------------------------------------------------------------------- */

    function createConfidence(
        memory
    ) {
        const value =
            percentage(
                memory.confidence
            );

        const wrapper =
            createElement(
                "div",
                "memory-confidence"
            );

        wrapper.setAttribute(
            "aria-label",
            `${value}% confidence`
        );

        const ring =
            createElement(
                "div",
                "memory-confidence__ring"
            );

        ring.style.setProperty(
            "--confidence",
            `${value}%`
        );

        const number =
            createElement(
                "strong",
                "memory-confidence__value",
                `${value}%`
            );

        const label =
            createElement(
                "span",
                "memory-confidence__label",
                "confidence"
            );

        ring.appendChild(
            number
        );

        wrapper.append(
            ring,
            label
        );

        return wrapper;
    }

    /* ----------------------------------------------------------------------
       Entities
       ---------------------------------------------------------------------- */

    function createEntities(
        memory
    ) {
        const wrapper =
            createElement(
                "div",
                "memory-card__entities"
            );

        if (
            !memory.entities.length
        ) {
            return wrapper;
        }

        memory.entities.forEach(
            (entity) => {
                const chip =
                    createChip(
                        "sparkles",
                        entity,
                        "memory-chip--entity"
                    );

                wrapper.appendChild(
                    chip
                );
            }
        );

        return wrapper;
    }

    /* ----------------------------------------------------------------------
       Why matched
       ---------------------------------------------------------------------- */

    function createWhyMatched(
        memory
    ) {
        const section =
            createElement(
                "div",
                "memory-card__why"
            );

        const header =
            createElement(
                "div",
                "memory-card__why-header"
            );

        const title =
            createElement(
                "span",
                "memory-card__why-title",
                "Why this matched?"
            );

        const icon =
            createIcon(
                "sparkles"
            );

        header.append(
            icon,
            title
        );

        section.appendChild(
            header
        );

        const reasons =
            createElement(
                "ul",
                "memory-card__why-list"
            );

        if (
            !memory.whyMatched.length
        ) {
            const item =
                createElement(
                    "li",
                    "memory-card__why-item"
                );

            item.appendChild(
                createIcon(
                    "check"
                )
            );

            item.appendChild(
                createElement(
                    "span",
                    "",
                    "Relevant semantic match"
                )
            );

            reasons.appendChild(
                item
            );
        } else {
            memory.whyMatched.forEach(
                (reason) => {
                    const item =
                        createElement(
                            "li",
                            "memory-card__why-item"
                        );

                    const reasonIcon =
                        reason.strength ===
                        "negative"
                            ? "minus"
                            : "check";

                    item.appendChild(
                        createIcon(
                            reasonIcon
                        )
                    );

                    item.appendChild(
                        createElement(
                            "span",
                            "",
                            reason.label
                        )
                    );

                    reasons.appendChild(
                        item
                    );
                }
            );
        }

        section.appendChild(
            reasons
        );

        return section;
    }

    /* ----------------------------------------------------------------------
       Memory Card
       ---------------------------------------------------------------------- */

    function createMemoryCard(
        memory,
        index
    ) {
        const card =
            createElement(
                "article",
                "memory-card"
            );

        card.dataset.memoryId =
            memory.id ||
            `memory-${index}`;

        card.dataset.index =
            String(index);

        card.setAttribute(
            "tabindex",
            "0"
        );

        card.setAttribute(
            "role",
            "article"
        );

        card.setAttribute(
            "aria-label",
            `${memory.title}. ${percentage(
                memory.confidence
            )} percent confidence.`
        );

        /*
         * Media
         */

        const image =
            createMemoryImage(
                memory
            );

        /*
         * Content
         */

        const content =
            createElement(
                "div",
                "memory-card__content"
            );

        const top =
            createElement(
                "div",
                "memory-card__top"
            );

        const category =
            createChip(
                "layers-3",
                memory.category,
                "memory-chip--category"
            );

        const confidence =
            createConfidence(
                memory
            );

        top.append(
            category,
            confidence
        );

        const title =
            createElement(
                "h3",
                "memory-card__title",
                memory.title
            );

        const summary =
            createElement(
                "p",
                "memory-card__summary",
                memory.summary
            );

        const metadata =
            createElement(
                "div",
                "memory-card__metadata"
            );

        const intent =
            createChip(
                "target",
                memory.intent,
                "memory-chip--intent"
            );

        metadata.appendChild(
            intent
        );

        const entities =
            createEntities(
                memory
            );

        content.append(
            top,
            title,
            summary,
            metadata,
            entities,
            createWhyMatched(
                memory
            )
        );

        /*
         * Footer
         */

        const footer =
            createElement(
                "div",
                "memory-card__footer"
            );

        const score =
            createElement(
                "span",
                "memory-card__score"
            );

        score.appendChild(
            createIcon(
                "activity"
            )
        );

        score.appendChild(
            createElement(
                "span",
                "",
                `${percentage(
                    memory.score
                )}% match`
            )
        );

        const action =
            createElement(
                "button",
                "memory-card__action",
                "View memory"
            );

        action.type =
            "button";

        action.setAttribute(
            "aria-label",
            `View ${memory.title}`
        );

        action.appendChild(
            createIcon(
                "arrow-up-right"
            )
        );

        footer.append(
            score,
            action
        );

        content.appendChild(
            footer
        );

        card.append(
            image,
            content
        );

        /*
         * Interaction
         */

        const open = () => {
            dispatchOpenMemory(
                memory
            );
        };

        card.addEventListener(
            "click",
            (event) => {
                if (
                    event.target instanceof
                    HTMLButtonElement ||
                    event.target.closest(
                        "button"
                    )
                ) {
                    return;
                }

                open();
            }
        );

        action.addEventListener(
            "click",
            (event) => {
                event.preventDefault();

                event.stopPropagation();

                open();
            }
        );

        card.addEventListener(
            "keydown",
            (event) => {
                if (
                    event.key === "Enter" ||
                    event.key === " "
                ) {
                    event.preventDefault();

                    open();
                }
            }
        );

        return card;
    }

    /* ----------------------------------------------------------------------
       Render results
       ---------------------------------------------------------------------- */

    function renderResults(
        results,
        total = null
    ) {
        if (!resolveElements()) {
            console.warn(
                "[MemoryOS Results] Results container not found."
            );

            return;
        }

        const normalizedResults =
            isArray(results)
                ? results
                    .map(
                        normalizeMemory
                    )
                    .filter(Boolean)
                : [];

        state.results =
            normalizedResults;

        state.total =
            total !== null &&
            Number.isFinite(
                Number(total)
            )
                ? Math.max(
                    0,
                    Number(total)
                )
                : normalizedResults.length;

        state.renderedAt =
            Date.now();

        state.error = null;

        setLoading(false);

        setError(false);

        setEmpty(
            normalizedResults.length ===
            0
        );

        renderCount(
            state.total
        );

        clearElement(
            state.container
        );

        if (
            !normalizedResults.length
        ) {
            renderInlineEmpty();

            return;
        }

        const fragment =
            document.createDocumentFragment();

        normalizedResults.forEach(
            (memory, index) => {
                fragment.appendChild(
                    createMemoryCard(
                        memory,
                        index
                    )
                );
            }
        );

        state.container.appendChild(
            fragment
        );

        refreshIcons(
            state.container
        );

        requestAnimationFrame(
            () => {
                revealCards();
            }
        );

        dispatch(
            CONFIG.events.resultsRendered,
            {
                results:
                    normalizedResults,

                total:
                    state.total,

                query:
                    state.query,

                renderedAt:
                    state.renderedAt,
            }
        );
    }

    /* ----------------------------------------------------------------------
       Empty state
       ---------------------------------------------------------------------- */

    function renderInlineEmpty() {
        if (!state.container) {
            return;
        }

        const empty =
            createElement(
                "div",
                "results-empty-state"
            );

        empty.setAttribute(
            "role",
            "status"
        );

        const iconWrap =
            createElement(
                "div",
                "results-empty-state__icon"
            );

        iconWrap.appendChild(
            createIcon(
                "search-x"
            )
        );

        const title =
            createElement(
                "h3",
                "results-empty-state__title",
                "No matching memories"
            );

        const description =
            createElement(
                "p",
                "results-empty-state__description",
                state.query
                    ? `MemoryOS couldn't find a strong match for “${truncate(
                        state.query,
                        100
                    )}”.`
                    : "Your memories will appear here once they are indexed."
            );

        const hint =
            createElement(
                "span",
                "results-empty-state__hint",
                "Try a more descriptive phrase or search by meaning."
            );

        empty.append(
            iconWrap,
            title,
            description,
            hint
        );

        state.container.appendChild(
            empty
        );

        refreshIcons(
            state.container
        );
    }

    /* ----------------------------------------------------------------------
       Loading
       ---------------------------------------------------------------------- */

    function showLoading(
        payload = {}
    ) {
        if (!resolveElements()) {
            return;
        }

        state.query =
            toSafeString(
                firstDefined(
                    payload.query,
                    state.query
                )
            );

        state.error = null;

        setError(false);

        setEmpty(false);

        setLoading(true);

        renderSkeletons(
            Number(
                payload.count
            ) || 6
        );

        renderCount(0);
    }

    /* ----------------------------------------------------------------------
       Error
       ---------------------------------------------------------------------- */

    function showError(
        payload = {}
    ) {
        if (!resolveElements()) {
            return;
        }

        state.loading =
            false;

        state.error =
            toSafeString(
                firstDefined(
                    payload.message,
                    payload.error,
                    "Unable to load memories."
                ),
                "Unable to load memories."
            );

        setLoading(false);

        setEmpty(false);

        setError(true);

        clearElement(
            state.container
        );

        const error =
            createElement(
                "div",
                "results-error-state"
            );

        error.setAttribute(
            "role",
            "alert"
        );

        const iconWrap =
            createElement(
                "div",
                "results-error-state__icon"
            );

        iconWrap.appendChild(
            createIcon(
                "cloud-off"
            )
        );

        const title =
            createElement(
                "h3",
                "results-error-state__title",
                "Search temporarily unavailable"
            );

        const message =
            createElement(
                "p",
                "results-error-state__message",
                "MemoryOS couldn't reach the search service. Your memories are safe."
            );

        const retry =
            createElement(
                "button",
                "results-error-state__action",
                "Try again"
            );

        retry.type =
            "button";

        retry.appendChild(
            createIcon(
                "rotate-cw"
            )
        );

        retry.addEventListener(
            "click",
            () => {
                dispatch(
                    CONFIG.events.refresh,
                    {
                        query:
                            state.query,

                        source:
                            "results-error-retry",
                    }
                );
            }
        );

        error.append(
            iconWrap,
            title,
            message,
            retry
        );

        state.container.appendChild(
            error
        );

        refreshIcons(
            state.container
        );
    }

    /* ----------------------------------------------------------------------
       Degraded notice
       ---------------------------------------------------------------------- */

    function showDegradedNotice(
        payload = {}
    ) {
        const message =
            toSafeString(
                firstDefined(
                    payload.message,
                    "Some memories could not be processed."
                )
            );

        if (!message) {
            return;
        }

        const notice =
            createElement(
                "div",
                "results-degraded-notice"
            );

        notice.setAttribute(
            "role",
            "status"
        );

        notice.appendChild(
            createIcon(
                "triangle-alert"
            )
        );

        notice.appendChild(
            createElement(
                "span",
                "",
                message
            )
        );

        if (state.container) {
            state.container.parentElement?.insertBefore(
                notice,
                state.container
            );
        }

        refreshIcons(
            notice
        );

        window.setTimeout(
            () => {
                notice.remove();
            },
            6500
        );
    }

    /* ----------------------------------------------------------------------
       Extract results
       ---------------------------------------------------------------------- */

    function extractResults(
        payload
    ) {
        if (!payload) {
            return {
                results: [],
                total: 0,
            };
        }

        if (isArray(payload)) {
            return {
                results: payload,
                total: payload.length,
            };
        }

        const results =
            firstDefined(
                payload.results,
                payload.memories,
                payload.items,
                payload.data?.results,
                payload.data?.memories,
                payload.data?.items
            );

        const total =
            firstDefined(
                payload.total,
                payload.count,
                payload.result_count,
                payload.data?.total,
                payload.data?.count,
                isArray(results)
                    ? results.length
                    : 0
            );

        return {
            results:
                isArray(results)
                    ? results
                    : [],

            total:
                Number(total) || 0,
        };
    }

    /* ----------------------------------------------------------------------
       Event dispatch
       ---------------------------------------------------------------------- */

    function dispatch(
        name,
        detail = {}
    ) {
        try {
            window.dispatchEvent(
                new CustomEvent(
                    name,
                    {
                        detail,
                    }
                )
            );
        } catch (error) {
            console.warn(
                `[MemoryOS Results] Event dispatch failed: ${name}`,
                error
            );
        }
    }

    function dispatchOpenMemory(
        memory
    ) {
        dispatch(
            CONFIG.events.openMemory,
            {
                memory:
                    memory.raw,

                normalizedMemory:
                    memory,
            }
        );
    }

    /* ----------------------------------------------------------------------
       Event handlers
       ---------------------------------------------------------------------- */

    function handleSearchLoading(
        event
    ) {
        const payload =
            isObject(
                event?.detail
            )
                ? event.detail
                : {};

        showLoading(
            payload
        );
    }

    function handleSearchComplete(
        event
    ) {
        const payload =
            isObject(
                event?.detail
            )
                ? event.detail
                : event?.detail ?? [];

        const extracted =
            extractResults(
                payload
            );

        const query =
            firstDefined(
                payload?.query,
                payload?.search_query,
                payload?.q,
                state.query
            );

        state.query =
            toSafeString(
                query
            );

        renderResults(
            extracted.results,
            extracted.total
        );

        const failed =
            firstDefined(
                payload?.failed,
                payload?.failed_count,
                payload?.errors
            );

        if (
            failed &&
            (
                Number(failed) > 0 ||
                isArray(failed)
            )
        ) {
            const failedCount =
                isArray(failed)
                    ? failed.length
                    : Number(failed);

            showDegradedNotice({
                message:
                    `${failedCount} result${
                        failedCount === 1
                            ? ""
                            : "s"
                    } could not be fully processed.`,
            });
        }

        if (
            payload?.degraded === true ||
            payload?.partial === true
        ) {
            showDegradedNotice({
                message:
                    toSafeString(
                        firstDefined(
                            payload?.degraded_message,
                            payload?.partial_message
                        ),
                        "Some results may be incomplete."
                    ),
            });
        }
    }

    function handleSearchEmpty(
        event
    ) {
        const payload =
            isObject(
                event?.detail
            )
                ? event.detail
                : {};

        state.query =
            toSafeString(
                firstDefined(
                    payload.query,
                    payload.search_query,
                    state.query
                )
            );

        renderResults(
            [],
            0
        );
    }

    function handleSearchError(
        event
    ) {
        const payload =
            isObject(
                event?.detail
            )
                ? event.detail
                : {};

        state.query =
            toSafeString(
                firstDefined(
                    payload.query,
                    payload.search_query,
                    state.query
                )
            );

        showError(
            payload
        );
    }

    function handleSearchCleared() {
        state.query = "";

        state.results = [];

        state.total = 0;

        state.error = null;

        if (!resolveElements()) {
            return;
        }

        setLoading(false);

        setError(false);

        setEmpty(false);

        clearElement(
            state.container
        );

        renderCount(0);
    }

    /* ----------------------------------------------------------------------
       Card reveal
       ---------------------------------------------------------------------- */

    function revealCards() {
        if (!state.container) {
            return;
        }

        const cards =
            state.container.querySelectorAll(
                ".memory-card:not(.memory-card--skeleton)"
            );

        cards.forEach(
            (card, index) => {
                card.style.setProperty(
                    "--result-index",
                    String(index)
                );

                card.classList.add(
                    "memory-card--visible"
                );
            }
        );
    }

    /* ----------------------------------------------------------------------
       Public API
       ---------------------------------------------------------------------- */

    const ResultsController = {
        init() {
            if (
                state.initialized
            ) {
                return this;
            }

            resolveElements();

            window.addEventListener(
                CONFIG.events.searchLoading,
                handleSearchLoading
            );

            window.addEventListener(
                CONFIG.events.searchComplete,
                handleSearchComplete
            );

            window.addEventListener(
                CONFIG.events.searchEmpty,
                handleSearchEmpty
            );

            window.addEventListener(
                CONFIG.events.searchError,
                handleSearchError
            );

            window.addEventListener(
                CONFIG.events.searchCleared,
                handleSearchCleared
            );

            state.initialized =
                true;

            return this;
        },

        render(
            results,
            total = null
        ) {
            renderResults(
                results,
                total
            );

            return this;
        },

        loading(
            payload = {}
        ) {
            showLoading(
                payload
            );

            return this;
        },

        error(
            payload = {}
        ) {
            showError(
                payload
            );

            return this;
        },

        clear() {
            handleSearchCleared();

            return this;
        },

        getResults() {
            return state.results.slice();
        },

        getState() {
            return {
                initialized:
                    state.initialized,

                loading:
                    state.loading,

                error:
                    state.error,

                query:
                    state.query,

                total:
                    state.total,

                renderedAt:
                    state.renderedAt,

                resultCount:
                    state.results.length,
            };
        },

        open(
            memoryOrIndex
        ) {
            if (
                typeof memoryOrIndex ===
                "number"
            ) {
                const memory =
                    state.results[
                        memoryOrIndex
                    ];

                if (memory) {
                    dispatchOpenMemory(
                        memory
                    );
                }

                return this;
            }

            const normalized =
                normalizeMemory(
                    memoryOrIndex
                );

            if (normalized) {
                dispatchOpenMemory(
                    normalized
                );
            }

            return this;
        },

        refresh() {
            dispatch(
                CONFIG.events.refresh,
                {
                    query:
                        state.query,

                    source:
                        "results-controller",
                }
            );

            return this;
        },

        destroy() {
            if (
                !state.initialized
            ) {
                return this;
            }

            window.removeEventListener(
                CONFIG.events.searchLoading,
                handleSearchLoading
            );

            window.removeEventListener(
                CONFIG.events.searchComplete,
                handleSearchComplete
            );

            window.removeEventListener(
                CONFIG.events.searchEmpty,
                handleSearchEmpty
            );

            window.removeEventListener(
                CONFIG.events.searchError,
                handleSearchError
            );

            window.removeEventListener(
                CONFIG.events.searchCleared,
                handleSearchCleared
            );

            state.initialized =
                false;

            return this;
        },
    };

    /* ----------------------------------------------------------------------
       Expose
       ---------------------------------------------------------------------- */

    window.MemoryOS.results =
        ResultsController;

    /* ----------------------------------------------------------------------
       Initialize
       ---------------------------------------------------------------------- */

    if (
        document.readyState ===
        "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            () =>
                ResultsController.init(),
            {
                once: true,
            }
        );
    } else {
        ResultsController.init();
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
            "%cMemoryOS Results%c initialized",
            "font-weight:700;",
            "font-weight:400;"
        );

        console.info(
            "[MemoryOS] Screenshot previews use:",
            `${resolveApiBase()}/upload/file/{memory_id}`
        );
    }
})();