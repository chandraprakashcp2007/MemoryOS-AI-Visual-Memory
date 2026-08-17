/**
 * ================================================================
 * MemoryOS — Search Controller
 * ================================================================
 *
 * Responsibilities:
 * - Natural-language memory search
 * - Search input handling
 * - Debounced suggestions
 * - Search button / Enter support
 * - Search loading state
 * - Recent searches
 * - Search history
 * - Result count + latency
 * - Empty/error/offline states
 * - Backend integration through existing api.js
 * - Safe event dispatching to results.js
 *
 * Business logic remains outside the UI.
 * ================================================================
 */

(() => {
    "use strict";

    /* ============================================================
       MEMORYOS NAMESPACE
       ============================================================ */

    const MemoryOS =
        window.MemoryOS ||
        (window.MemoryOS = {});


    /* ============================================================
       CONFIGURATION
       ============================================================ */

    const SEARCH_CONFIG = {
        debounceDelay: 280,

        minimumQueryLength: 1,

        maximumQueryLength: 500,

        recentSearchLimit: 8,

        suggestionLimit: 6,

        storageKey:
            "memoryos-recent-searches",

        loadingClass:
            "is-searching",

        activeClass:
            "is-active",

        requestTimeout: 30000,
    };


    /* ============================================================
       STATE
       ============================================================ */

    const state = {
        query: "",

        lastQuery: "",

        isSearching: false,

        hasSearched: false,

        resultsCount: 0,

        latency: 0,

        requestId: 0,

        debounceTimer: null,

        abortController: null,

        recentSearches: [],

        suggestionsVisible: false,
    };


    /* ============================================================
       DOM CACHE
       ============================================================ */

    const elements = {
        searchInput:
            document.querySelector(
                "#searchInput"
            ) ||
            document.querySelector(
                "#heroSearchInput"
            ) ||
            document.querySelector(
                "[data-search-input]"
            ) ||
            document.querySelector(
                "input[type='search']"
            ),

        searchForm:
            document.querySelector(
                "#searchForm"
            ) ||
            document.querySelector(
                "[data-search-form]"
            ),

        searchButton:
            document.querySelector(
                "#searchButton"
            ) ||
            document.querySelector(
                "[data-search-button]"
            ) ||
            document.querySelector(
                "[data-search-submit]"
            ),

        suggestions:
            document.querySelector(
                "#searchSuggestions"
            ) ||
            document.querySelector(
                "[data-search-suggestions]"
            ),

        suggestionItems:
            document.querySelector(
                "[data-suggestion-list]"
            ),

        recentSearches:
            document.querySelector(
                "#recentSearches"
            ) ||
            document.querySelector(
                "[data-recent-searches]"
            ),

        clearSearch:
            document.querySelector(
                "#clearSearch"
            ) ||
            document.querySelector(
                "[data-clear-search]"
            ),

        searchStatus:
            document.querySelector(
                "#searchStatus"
            ) ||
            document.querySelector(
                "[data-search-status]"
            ),

        resultCount:
            document.querySelector(
                "#searchResultCount"
            ) ||
            document.querySelector(
                "[data-search-result-count]"
            ),

        searchLatency:
            document.querySelector(
                "#searchLatency"
            ) ||
            document.querySelector(
                "[data-search-latency]"
            ),

        searchQueryDisplay:
            document.querySelector(
                "#searchQuery"
            ) ||
            document.querySelector(
                "[data-search-query]"
            ),

        searchEmpty:
            document.querySelector(
                "#searchEmpty"
            ) ||
            document.querySelector(
                "[data-search-empty]"
            ),

        searchError:
            document.querySelector(
                "#searchError"
            ) ||
            document.querySelector(
                "[data-search-error]"
            ),

        searchLoading:
            document.querySelector(
                "#searchLoading"
            ) ||
            document.querySelector(
                "[data-search-loading]"
            ),

        resultsContainer:
            document.querySelector(
                "#memoryGrid"
            ) ||
            document.querySelector(
                "#results"
            ) ||
            document.querySelector(
                "[data-results]"
            ),
    };


    /* ============================================================
       SAFE HELPERS
       ============================================================ */

    function safeString(
        value
    ) {
        return String(
            value ?? ""
        );
    }


    function normalizeQuery(
        value
    ) {
        return safeString(value)
            .replace(/\s+/g, " ")
            .trim()
            .slice(
                0,
                SEARCH_CONFIG.maximumQueryLength
            );
    }


    function isValidQuery(
        query
    ) {
        return (
            query.length >=
            SEARCH_CONFIG.minimumQueryLength
        );
    }


    function prefersReducedMotion() {
        return window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;
    }


    /* ============================================================
       STORAGE
       ============================================================ */

    function loadRecentSearches() {
        try {
            const raw =
                localStorage.getItem(
                    SEARCH_CONFIG.storageKey
                );

            if (!raw) {
                state.recentSearches = [];
                return;
            }

            const parsed =
                JSON.parse(raw);

            if (
                !Array.isArray(
                    parsed
                )
            ) {
                state.recentSearches = [];
                return;
            }

            state.recentSearches =
                parsed
                    .filter(
                        (item) =>
                            typeof item ===
                            "string"
                    )
                    .map(
                        (item) =>
                            normalizeQuery(
                                item
                            )
                    )
                    .filter(Boolean)
                    .slice(
                        0,
                        SEARCH_CONFIG.recentSearchLimit
                    );
        } catch {
            state.recentSearches = [];
        }
    }


    function saveRecentSearches() {
        try {
            localStorage.setItem(
                SEARCH_CONFIG.storageKey,
                JSON.stringify(
                    state.recentSearches
                )
            );
        } catch {
            /*
             * Storage failure should never
             * break search functionality.
             */
        }
    }


    function addRecentSearch(
        query
    ) {
        const normalized =
            normalizeQuery(query);

        if (!normalized) {
            return;
        }

        state.recentSearches =
            [
                normalized,
                ...state.recentSearches.filter(
                    (item) =>
                        item.toLowerCase() !==
                        normalized.toLowerCase()
                ),
            ].slice(
                0,
                SEARCH_CONFIG.recentSearchLimit
            );

        saveRecentSearches();

        renderRecentSearches();
    }


    function clearRecentSearches() {
        state.recentSearches = [];

        try {
            localStorage.removeItem(
                SEARCH_CONFIG.storageKey
            );
        } catch {
            // Ignore storage failures.
        }

        renderRecentSearches();
    }


    /* ============================================================
       API RESOLUTION
       ============================================================ */

    function resolveApiClient() {
        /*
         * Supports multiple clean API client
         * naming conventions without creating
         * another API implementation here.
         */

        if (
            MemoryOS.api &&
            typeof MemoryOS.api.search ===
                "function"
        ) {
            return MemoryOS.api;
        }

        if (
            MemoryOS.api &&
            typeof MemoryOS.api.searchMemories ===
                "function"
        ) {
            return {
                search:
                    MemoryOS.api
                        .searchMemories,
            };
        }

        if (
            MemoryOS.API &&
            typeof MemoryOS.API.search ===
                "function"
        ) {
            return MemoryOS.API;
        }

        return null;
    }


    /* ============================================================
       API SEARCH
       ============================================================ */

    async function requestSearch(
        query,
        requestId
    ) {
        const api =
            resolveApiClient();

        if (!api) {
            throw new Error(
                "Search service is unavailable."
            );
        }

        /*
         * Cancel previous request.
         */
        if (
            state.abortController
        ) {
            state.abortController.abort();
        }

        state.abortController =
            new AbortController();

        const timeout =
            window.setTimeout(
                () => {
                    state.abortController?.abort();
                },
                SEARCH_CONFIG.requestTimeout
            );

        const startedAt =
            performance.now();

        try {
            let response;

            /*
             * Preferred:
             * existing api.js search method.
             */
            try {
                response =
                    await api.search(
                        query,
                        {
                            signal:
                                state
                                    .abortController
                                    .signal,
                        }
                    );
            } catch (
                firstError
            ) {
                /*
                 * Some API clients may expect
                 * an object instead.
                 */
                if (
                    requestId !==
                    state.requestId
                ) {
                    return null;
                }

                response =
                    await api.search({
                        query,
                        signal:
                            state
                                .abortController
                                .signal,
                    });
            }

            if (
                requestId !==
                state.requestId
            ) {
                return null;
            }

            const latency =
                Math.round(
                    performance.now() -
                        startedAt
                );

            return {
                response,

                latency,
            };
        } finally {
            window.clearTimeout(
                timeout
            );
        }
    }


    /* ============================================================
       RESPONSE NORMALIZATION
       ============================================================ */

    function normalizeSearchResponse(
        payload
    ) {
        if (!payload) {
            return {
                results: [],
                total: 0,
                query: state.query,
                latency: 0,
            };
        }

        /*
         * Support common FastAPI response shapes.
         */
        const results =
            Array.isArray(
                payload
            )
                ? payload
                : Array.isArray(
                      payload.results
                  )
                ? payload.results
                : Array.isArray(
                      payload.memories
                  )
                ? payload.memories
                : Array.isArray(
                      payload.items
                  )
                ? payload.items
                : [];

        const total =
            Number.isFinite(
                Number(
                    payload.total
                )
            )
                ? Number(
                      payload.total
                  )
                : results.length;

        return {
            ...payload,

            results,

            total,

            query:
                payload.query ||
                state.query,
        };
    }


    /* ============================================================
       SEARCH EXECUTION
       ============================================================ */

    async function performSearch(
        rawQuery,
        options = {}
    ) {
        const query =
            normalizeQuery(
                rawQuery
            );

        if (
            !isValidQuery(query)
        ) {
            showValidationState();
            return;
        }

        /*
         * Avoid duplicate request unless
         * explicitly requested.
         */
        if (
            !options.force &&
            query.toLowerCase() ===
                state.lastQuery.toLowerCase() &&
            state.hasSearched
        ) {
            return;
        }

        state.query =
            query;

        state.lastQuery =
            query;

        state.hasSearched =
            true;

        state.requestId += 1;

        const requestId =
            state.requestId;

        state.isSearching =
            true;

        hideSuggestions();

        clearSearchError();

        clearSearchEmpty();

        updateSearchUI();

        dispatch(
            "memoryos:search-loading",
            {
                query,
            }
        );

        try {
            const result =
                await requestSearch(
                    query,
                    requestId
                );

            /*
             * Request was superseded.
             */
            if (
                !result ||
                requestId !==
                    state.requestId
            ) {
                return;
            }

            const normalized =
                normalizeSearchResponse(
                    result.response
                );

            state.resultsCount =
                normalized.total;

            state.latency =
                result.latency;

            state.isSearching =
                false;

            updateSearchUI();

            addRecentSearch(
                query
            );

            dispatch(
                "memoryos:search-complete",
                {
                    query,

                    results:
                        normalized.results,

                    total:
                        normalized.total,

                    latency:
                        result.latency,

                    response:
                        normalized,
                }
            );

            /*
             * Empty result state.
             */
            if (
                normalized.results.length ===
                    0 ||
                normalized.total ===
                    0
            ) {
                showSearchEmpty(
                    query
                );
            }

            /*
             * Update result UI if a
             * results module is present.
             */
            if (
                MemoryOS.results &&
                typeof MemoryOS.results
                    .render ===
                    "function"
            ) {
                MemoryOS.results.render(
                    normalized.results,
                    {
                        query,

                        total:
                            normalized.total,

                        latency:
                            result.latency,
                    }
                );
            }
        } catch (error) {
            /*
             * Ignore intentionally aborted requests.
             */
            if (
                error?.name ===
                    "AbortError" ||
                requestId !==
                    state.requestId
            ) {
                return;
            }

            state.isSearching =
                false;

            updateSearchUI();

            const message =
                getSafeErrorMessage(
                    error
                );

            showSearchError(
                message
            );

            dispatch(
                "memoryos:search-error",
                {
                    query,

                    message,

                    error:
                        error instanceof
                        Error
                            ? error
                            : null,
                }
            );
        }
    }


    /* ============================================================
       ERROR HANDLING
       ============================================================ */

    function getSafeErrorMessage(
        error
    ) {
        if (
            !navigator.onLine
        ) {
            return "You appear to be offline. Check your connection and try again.";
        }

        const message =
            safeString(
                error?.message
            ).toLowerCase();

        if (
            message.includes(
                "abort"
            )
        ) {
            return "Search was cancelled.";
        }

        if (
            message.includes(
                "timeout"
            )
        ) {
            return "Search is taking too long. Please try again.";
        }

        if (
            message.includes(
                "network"
            ) ||
            message.includes(
                "failed to fetch"
            )
        ) {
            return "MemoryOS could not reach the server. Make sure the backend is running.";
        }

        return "Something went wrong while searching. Please try again.";
    }


    /* ============================================================
       DEBOUNCE
       ============================================================ */

    function scheduleSearch(
        query
    ) {
        if (
            state.debounceTimer
        ) {
            window.clearTimeout(
                state.debounceTimer
            );
        }

        const normalized =
            normalizeQuery(query);

        if (
            !normalized
        ) {
            state.query = "";

            hideSuggestions();

            return;
        }

        state.debounceTimer =
            window.setTimeout(
                () => {
                    /*
                     * Debounce is primarily useful
                     * for suggestions.
                     *
                     * Actual search remains
                     * Enter/button driven so we
                     * don't spam the backend.
                     */
                    renderSuggestions(
                        normalized
                    );
                },
                SEARCH_CONFIG.debounceDelay
            );
    }


    /* ============================================================
       SUGGESTIONS
       ============================================================ */

    function getDefaultSuggestions() {
        return [
            "the shoes I wanted to buy",

            "my Python study notes",

            "receipt from my online order",

            "travel information I saved",

            "the address from my screenshot",

            "that coding error I saved",
        ];
    }


    function buildSuggestions(
        query
    ) {
        const normalized =
            normalizeQuery(
                query
            ).toLowerCase();

        const recent =
            state.recentSearches
                .filter(
                    (item) =>
                        item
                            .toLowerCase()
                            .includes(
                                normalized
                            )
                );

        const defaults =
            getDefaultSuggestions()
                .filter(
                    (item) =>
                        item
                            .toLowerCase()
                            .includes(
                                normalized
                            )
                );

        const combined =
            [
                ...recent,
                ...defaults,
            ];

        const unique =
            [];

        const seen =
            new Set();

        combined.forEach(
            (item) => {
                const key =
                    item.toLowerCase();

                if (
                    seen.has(key)
                ) {
                    return;
                }

                seen.add(key);

                unique.push(item);
            }
        );

        return unique.slice(
            0,
            SEARCH_CONFIG.suggestionLimit
        );
    }


    function renderSuggestions(
        query
    ) {
        const container =
            elements.suggestions;

        if (!container) {
            return;
        }

        const suggestions =
            buildSuggestions(
                query
            );

        const list =
            elements.suggestionItems ||
            container;

        list.replaceChildren();

        if (
            suggestions.length ===
            0
        ) {
            hideSuggestions();
            return;
        }

        const fragment =
            document.createDocumentFragment();

        suggestions.forEach(
            (suggestion) => {
                const button =
                    document.createElement(
                        "button"
                    );

                button.type =
                    "button";

                button.className =
                    "search-suggestion";

                button.dataset.query =
                    suggestion;

                button.setAttribute(
                    "role",
                    "option"
                );

                const icon =
                    document.createElement(
                        "span"
                    );

                icon.className =
                    "search-suggestion__icon";

                icon.setAttribute(
                    "aria-hidden",
                    "true"
                );

                icon.textContent =
                    "⌕";

                const text =
                    document.createElement(
                        "span"
                    );

                text.className =
                    "search-suggestion__text";

                text.textContent =
                    suggestion;

                button.appendChild(
                    icon
                );

                button.appendChild(
                    text
                );

                fragment.appendChild(
                    button
                );
            }
        );

        list.appendChild(
            fragment
        );

        container.hidden =
            false;

        container.classList.add(
            "is-visible"
        );

        state.suggestionsVisible =
            true;
    }


    function hideSuggestions() {
        if (
            !elements.suggestions
        ) {
            return;
        }

        elements.suggestions.classList.remove(
            "is-visible"
        );

        elements.suggestions.hidden =
            true;

        state.suggestionsVisible =
            false;
    }


    /* ============================================================
       RECENT SEARCHES UI
       ============================================================ */

    function renderRecentSearches() {
        const container =
            elements.recentSearches;

        if (!container) {
            return;
        }

        container.replaceChildren();

        if (
            state.recentSearches.length ===
            0
        ) {
            container.hidden =
                true;

            return;
        }

        container.hidden =
            false;

        const fragment =
            document.createDocumentFragment();

        state.recentSearches.forEach(
            (query) => {
                const button =
                    document.createElement(
                        "button"
                    );

                button.type =
                    "button";

                button.className =
                    "recent-search";

                button.dataset.query =
                    query;

                button.textContent =
                    query;

                fragment.appendChild(
                    button
                );
            }
        );

        container.appendChild(
            fragment
        );
    }


    /* ============================================================
       UI STATE
       ============================================================ */

    function updateSearchUI() {
        const searching =
            state.isSearching;

        elements.body?.classList.toggle(
            SEARCH_CONFIG.loadingClass,
            searching
        );

        elements.searchInput?.classList.toggle(
            "is-loading",
            searching
        );

        elements.searchButton?.classList.toggle(
            "is-loading",
            searching
        );

        if (
            elements.searchButton
        ) {
            elements.searchButton.disabled =
                searching;
        }

        if (
            elements.searchInput
        ) {
            elements.searchInput.setAttribute(
                "aria-busy",
                String(searching)
            );
        }

        if (
            elements.searchLoading
        ) {
            elements.searchLoading.hidden =
                !searching;
        }

        updateResultMeta();

        updateQueryDisplay();
    }


    function updateResultMeta() {
        if (
            elements.resultCount
        ) {
            elements.resultCount.textContent =
                String(
                    state.resultsCount
                );
        }

        if (
            elements.searchLatency
        ) {
            elements.searchLatency.textContent =
                state.latency > 0
                    ? `${state.latency} ms`
                    : "—";
        }
    }


    function updateQueryDisplay() {
        if (
            elements.searchQueryDisplay
        ) {
            elements.searchQueryDisplay.textContent =
                state.query ||
                "Search your memories";
        }
    }


    /* ============================================================
       EMPTY STATE
       ============================================================ */

    function showSearchEmpty(
        query
    ) {
        if (
            elements.searchEmpty
        ) {
            elements.searchEmpty.hidden =
                false;

            const message =
                elements.searchEmpty.querySelector(
                    "[data-empty-query]"
                );

            if (message) {
                message.textContent =
                    query;
            }
        }

        elements.body?.classList.add(
            "has-no-search-results"
        );

        dispatch(
            "memoryos:search-empty",
            {
                query,
            }
        );
    }


    function clearSearchEmpty() {
        if (
            elements.searchEmpty
        ) {
            elements.searchEmpty.hidden =
                true;
        }

        elements.body?.classList.remove(
            "has-no-search-results"
        );
    }


    /* ============================================================
       ERROR STATE
       ============================================================ */

    function showSearchError(
        message
    ) {
        if (
            elements.searchError
        ) {
            elements.searchError.hidden =
                false;

            const messageElement =
                elements.searchError.querySelector(
                    "[data-error-message]"
                );

            if (
                messageElement
            ) {
                messageElement.textContent =
                    safeString(
                        message
                    );
            }
        }

        elements.body?.classList.add(
            "has-search-error"
        );
    }


    function clearSearchError() {
        if (
            elements.searchError
        ) {
            elements.searchError.hidden =
                true;
        }

        elements.body?.classList.remove(
            "has-search-error"
        );
    }


    /* ============================================================
       VALIDATION STATE
       ============================================================ */

    function showValidationState() {
        if (
            elements.searchInput
        ) {
            elements.searchInput.setAttribute(
                "aria-invalid",
                "true"
            );
        }

        showSearchError(
            "Type something to search your memories."
        );
    }


    /* ============================================================
       CLEAR SEARCH
       ============================================================ */

    function clearSearch() {
        if (
            elements.searchInput
        ) {
            elements.searchInput.value =
                "";

            elements.searchInput.setAttribute(
                "aria-invalid",
                "false"
            );

            elements.searchInput.focus();
        }

        state.query = "";

        clearSearchError();

        clearSearchEmpty();

        hideSuggestions();

        updateSearchUI();

        dispatch(
            "memoryos:search-cleared"
        );
    }


    /* ============================================================
       EVENT DISPATCHER
       ============================================================ */

    function dispatch(
        eventName,
        detail = {}
    ) {
        window.dispatchEvent(
            new CustomEvent(
                eventName,
                {
                    detail,
                }
            )
        );
    }


    /* ============================================================
       FORM EVENTS
       ============================================================ */

    function bindSearchForm() {
        elements.searchForm?.addEventListener(
            "submit",
            (event) => {
                event.preventDefault();

                performSearch(
                    elements.searchInput
                        ?.value || ""
                );
            }
        );


        elements.searchButton?.addEventListener(
            "click",
            (event) => {
                event.preventDefault();

                performSearch(
                    elements.searchInput
                        ?.value || ""
                );
            }
        );
    }


    /* ============================================================
       INPUT EVENTS
       ============================================================ */

    function bindSearchInput() {
        if (
            !elements.searchInput
        ) {
            return;
        }

        elements.searchInput.addEventListener(
            "input",
            (event) => {
                const query =
                    normalizeQuery(
                        event.target
                            .value
                    );

                state.query =
                    query;

                event.target.setAttribute(
                    "aria-invalid",
                    "false"
                );

                clearSearchError();

                scheduleSearch(
                    query
                );

                /*
                 * Show recent/default
                 * suggestions while typing.
                 */
                if (query) {
                    renderSuggestions(
                        query
                    );
                } else {
                    hideSuggestions();
                }
            }
        );


        elements.searchInput.addEventListener(
            "keydown",
            (event) => {
                /*
                 * Enter = search.
                 */
                if (
                    event.key ===
                    "Enter"
                ) {
                    event.preventDefault();

                    performSearch(
                        event.target
                            .value
                    );

                    return;
                }

                /*
                 * Escape = hide suggestions.
                 */
                if (
                    event.key ===
                    "Escape"
                ) {
                    hideSuggestions();
                }
            }
        );


        elements.searchInput.addEventListener(
            "focus",
            () => {
                const query =
                    normalizeQuery(
                        elements.searchInput
                            .value
                    );

                if (query) {
                    renderSuggestions(
                        query
                    );
                }
            }
        );
    }


    /* ============================================================
       SUGGESTION EVENTS
       ============================================================ */

    function bindSuggestionEvents() {
        document.addEventListener(
            "click",
            (event) => {
                const suggestion =
                    event.target.closest(
                        "[data-query]"
                    );

                if (
                    !suggestion
                ) {
                    return;
                }

                /*
                 * Only handle search-specific
                 * suggestion elements.
                 */
                if (
                    !suggestion.classList.contains(
                        "search-suggestion"
                    ) &&
                    !suggestion.classList.contains(
                        "recent-search"
                    )
                ) {
                    return;
                }

                const query =
                    normalizeQuery(
                        suggestion.dataset
                            .query
                    );

                if (
                    !query
                ) {
                    return;
                }

                if (
                    elements.searchInput
                ) {
                    elements.searchInput.value =
                        query;
                }

                hideSuggestions();

                performSearch(
                    query,
                    {
                        force: true,
                    }
                );
            }
        );
    }


    /* ============================================================
       OUTSIDE CLICK
       ============================================================ */

    function bindOutsideClick() {
        document.addEventListener(
            "click",
            (event) => {
                if (
                    !state.suggestionsVisible
                ) {
                    return;
                }

                const searchArea =
                    event.target.closest(
                        "[data-search-container], .search-container, #searchForm"
                    );

                if (
                    !searchArea
                ) {
                    hideSuggestions();
                }
            }
        );
    }


    /* ============================================================
       CLEAR BUTTON
       ============================================================ */

    function bindClearButton() {
        elements.clearSearch?.addEventListener(
            "click",
            (event) => {
                event.preventDefault();

                clearSearch();
            }
        );
    }


    /* ============================================================
       GLOBAL SEARCH EVENTS
       ============================================================ */

    function bindGlobalSearchEvents() {
        /*
         * App.js can ask search.js to focus
         * the search field.
         */
        window.addEventListener(
            "memoryos:focus-search",
            () => {
                elements.searchInput?.focus();
            }
        );


        /*
         * Search can be requested from
         * command palette / other modules.
         */
        window.addEventListener(
            "memoryos:perform-search",
            (event) => {
                const query =
                    event.detail?.query;

                if (
                    typeof query !==
                    "string"
                ) {
                    return;
                }

                if (
                    elements.searchInput
                ) {
                    elements.searchInput.value =
                        query;
                }

                performSearch(
                    query,
                    {
                        force: true,
                    }
                );
            }
        );


        /*
         * Refresh current search after
         * uploads or memory changes.
         */
        window.addEventListener(
            "memoryos:refresh",
            () => {
                if (
                    state.lastQuery
                ) {
                    performSearch(
                        state.lastQuery,
                        {
                            force: true,
                        }
                    );
                }
            }
        );
    }


    /* ============================================================
       KEYBOARD SEARCH SHORTCUT
       ============================================================ */

    function bindKeyboardShortcut() {
        document.addEventListener(
            "keydown",
            (event) => {
                const modifier =
                    event.ctrlKey ||
                    event.metaKey;

                /*
                 * Ctrl/Cmd + /
                 *
                 * Focus search.
                 */
                if (
                    modifier &&
                    event.key === "/"
                ) {
                    event.preventDefault();

                    elements.searchInput?.focus();
                }
            }
        );
    }


    /* ============================================================
       INIT
       ============================================================ */

    function initialize() {
        loadRecentSearches();

        renderRecentSearches();

        bindSearchForm();

        bindSearchInput();

        bindSuggestionEvents();

        bindOutsideClick();

        bindClearButton();

        bindGlobalSearchEvents();

        bindKeyboardShortcut();

        updateSearchUI();

        dispatch(
            "memoryos:search-ready"
        );
    }


    /* ============================================================
       PUBLIC API
       ============================================================ */

    MemoryOS.search = {
        search: performSearch,

        clear: clearSearch,

        focus: () => {
            elements.searchInput?.focus();
        },

        getQuery: () =>
            state.query,

        getLastQuery: () =>
            state.lastQuery,

        getState: () => ({
            ...state,
            recentSearches: [
                ...state.recentSearches,
            ],
        }),

        clearHistory:
            clearRecentSearches,
    };


    /* ============================================================
       START
       ============================================================ */

    if (
        document.readyState ===
        "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {
                once: true,
            }
        );
    } else {
        initialize();
    }

})();