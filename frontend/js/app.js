/**
 * ================================================================
 * MemoryOS — Global UI Application Controller
 * ================================================================
 *
 * Controls the premium frontend experience:
 *
 * - Theme
 * - Navigation
 * - Command palette
 * - Memory drawer
 * - Filter panel
 * - Toast notifications
 * - Mobile navigation
 * - Global keyboard shortcuts
 * - Section scrolling
 * - Scroll-aware header
 * - Overlay management
 * - UI state synchronization
 *
 * This file intentionally DOES NOT contain:
 *
 * - API business logic
 * - Upload business logic
 * - Search business logic
 * - Result rendering logic
 * - Statistics fetching logic
 *
 * Those responsibilities remain modular.
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
       STATE
       ============================================================ */

    const state = {
        theme:
            localStorage.getItem(
                "memoryos-theme"
            ) || "dark",

        commandPaletteOpen: false,

        drawerOpen: false,

        filterPanelOpen: false,

        mobileNavigationOpen: false,

        activeSection: "home",

        toastCounter: 0,

        initialized: false,
    };


    /* ============================================================
       DOM CACHE
       ============================================================ */

    const elements = {
        body:
            document.body,

        html:
            document.documentElement,

        header:
            document.querySelector(
                "header"
            ),

        nav:
            document.querySelector(
                "nav"
            ),

        mobileMenu:
            document.querySelector(
                "#mobileMenu, [data-mobile-menu]"
            ),

        mobileToggle:
            document.querySelector(
                "#mobileMenuToggle, [data-mobile-menu-toggle]"
            ),

        mobileClose:
            document.querySelector(
                "#mobileMenuClose, [data-mobile-menu-close]"
            ),

        themeToggle:
            document.querySelector(
                "#themeToggle, [data-theme-toggle]"
            ),

        themeIcon:
            document.querySelector(
                "#themeIcon, [data-theme-icon]"
            ),

        commandPalette:
            document.querySelector(
                "#commandPalette, [data-command-palette]"
            ),

        commandInput:
            document.querySelector(
                "#commandInput, [data-command-input]"
            ),

        commandBackdrop:
            document.querySelector(
                "#commandPaletteBackdrop, [data-command-backdrop]"
            ),

        commandClose:
            document.querySelector(
                "#commandPaletteClose, [data-command-close]"
            ),

        drawer:
            document.querySelector(
                "#memoryDrawer, [data-memory-drawer]"
            ),

        drawerBackdrop:
            document.querySelector(
                "#drawerBackdrop, [data-drawer-backdrop]"
            ),

        drawerClose:
            document.querySelector(
                "#drawerClose, [data-drawer-close]"
            ),

        filterPanel:
            document.querySelector(
                "#filterPanel, [data-filter-panel]"
            ),

        filterBackdrop:
            document.querySelector(
                "#filterBackdrop, [data-filter-backdrop]"
            ),

        filterToggle:
            document.querySelector(
                "#filterToggle, [data-filter-toggle]"
            ),

        filterClose:
            document.querySelector(
                "#filterClose, [data-filter-close]"
            ),

        toastContainer:
            document.querySelector(
                "#toastContainer, [data-toast-container]"
            ),

        uploadButtons:
            document.querySelectorAll(
                "[data-upload], #uploadAction, #heroUpload"
            ),

        searchButtons:
            document.querySelectorAll(
                "[data-search-action]"
            ),

        navLinks:
            document.querySelectorAll(
                "[data-nav], nav a[href^='#']"
            ),

        sections:
            document.querySelectorAll(
                "main section[id], section[id]"
            ),

        scrollTop:
            document.querySelector(
                "#scrollTop, [data-scroll-top]"
            ),
    };


    /* ============================================================
       UTILITY — SAFE TEXT
       ============================================================ */

    function safeText(
        value
    ) {
        return String(
            value ?? ""
        );
    }


    /* ============================================================
       UTILITY — REDUCED MOTION
       ============================================================ */

    function prefersReducedMotion() {
        return window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches;
    }


    /* ============================================================
       UTILITY — FOCUS
       ============================================================ */

    function focusElement(
        element
    ) {
        if (!element) {
            return;
        }

        window.requestAnimationFrame(
            () => {
                try {
                    element.focus({
                        preventScroll:
                            true,
                    });
                } catch {
                    element.focus();
                }
            }
        );
    }


    /* ============================================================
       THEME SYSTEM
       ============================================================ */

    function applyTheme(
        theme
    ) {
        const validThemes = [
            "dark",
            "light",
        ];

        const nextTheme =
            validThemes.includes(
                theme
            )
                ? theme
                : "dark";

        state.theme =
            nextTheme;

        elements.html.dataset.theme =
            nextTheme;

        elements.html.classList.toggle(
            "theme-dark",
            nextTheme === "dark"
        );

        elements.html.classList.toggle(
            "theme-light",
            nextTheme === "light"
        );

        localStorage.setItem(
            "memoryos-theme",
            nextTheme
        );

        updateThemeButton();

        window.dispatchEvent(
            new CustomEvent(
                "memoryos:theme-change",
                {
                    detail: {
                        theme:
                            nextTheme,
                    },
                }
            )
        );
    }


    function updateThemeButton() {
        if (
            !elements.themeToggle
        ) {
            return;
        }

        const isDark =
            state.theme === "dark";

        elements.themeToggle.setAttribute(
            "aria-label",
            isDark
                ? "Switch to light theme"
                : "Switch to dark theme"
        );

        elements.themeToggle.setAttribute(
            "aria-pressed",
            String(isDark)
        );

        if (
            elements.themeIcon
        ) {
            elements.themeIcon.textContent =
                isDark
                    ? "☼"
                    : "☾";
        }
    }


    function toggleTheme() {
        applyTheme(
            state.theme ===
                "dark"
                ? "light"
                : "dark"
        );
    }


    /* ============================================================
       BODY SCROLL LOCK
       ============================================================ */

    function updateBodyLock() {
        const shouldLock =
            state.commandPaletteOpen ||
            state.drawerOpen ||
            state.filterPanelOpen ||
            state.mobileNavigationOpen;

        elements.body.classList.toggle(
            "is-ui-locked",
            shouldLock
        );
    }


    /* ============================================================
       MOBILE NAVIGATION
       ============================================================ */

    function openMobileNavigation() {
        if (
            !elements.mobileMenu
        ) {
            return;
        }

        state.mobileNavigationOpen =
            true;

        elements.mobileMenu.hidden =
            false;

        elements.mobileMenu.classList.add(
            "is-open"
        );

        elements.mobileMenu.setAttribute(
            "aria-hidden",
            "false"
        );

        if (
            elements.mobileToggle
        ) {
            elements.mobileToggle.setAttribute(
                "aria-expanded",
                "true"
            );
        }

        updateBodyLock();
    }


    function closeMobileNavigation() {
        if (
            !elements.mobileMenu
        ) {
            return;
        }

        state.mobileNavigationOpen =
            false;

        elements.mobileMenu.classList.remove(
            "is-open"
        );

        elements.mobileMenu.setAttribute(
            "aria-hidden",
            "true"
        );

        /*
         * Keep hidden attribute until the CSS
         * transition has completed.
         */
        if (
            prefersReducedMotion()
        ) {
            elements.mobileMenu.hidden =
                true;
        } else {
            window.setTimeout(
                () => {
                    if (
                        !state.mobileNavigationOpen
                    ) {
                        elements.mobileMenu.hidden =
                            true;
                    }
                },
                280
            );
        }

        if (
            elements.mobileToggle
        ) {
            elements.mobileToggle.setAttribute(
                "aria-expanded",
                "false"
            );
        }

        updateBodyLock();
    }


    function toggleMobileNavigation() {
        if (
            state.mobileNavigationOpen
        ) {
            closeMobileNavigation();
        } else {
            openMobileNavigation();
        }
    }


    /* ============================================================
       COMMAND PALETTE
       ============================================================ */

    function openCommandPalette() {
        if (
            !elements.commandPalette
        ) {
            return;
        }

        state.commandPaletteOpen =
            true;

        elements.commandPalette.hidden =
            false;

        elements.commandPalette.classList.add(
            "is-open"
        );

        elements.commandPalette.setAttribute(
            "aria-hidden",
            "false"
        );

        updateBodyLock();

        focusElement(
            elements.commandInput
        );
    }


    function closeCommandPalette() {
        if (
            !elements.commandPalette
        ) {
            return;
        }

        state.commandPaletteOpen =
            false;

        elements.commandPalette.classList.remove(
            "is-open"
        );

        elements.commandPalette.setAttribute(
            "aria-hidden",
            "true"
        );

        if (
            prefersReducedMotion()
        ) {
            elements.commandPalette.hidden =
                true;
        } else {
            window.setTimeout(
                () => {
                    if (
                        !state.commandPaletteOpen
                    ) {
                        elements.commandPalette.hidden =
                            true;
                    }
                },
                220
            );
        }

        updateBodyLock();
    }


    function toggleCommandPalette() {
        if (
            state.commandPaletteOpen
        ) {
            closeCommandPalette();
        } else {
            openCommandPalette();
        }
    }


    /* ============================================================
       COMMAND EXECUTION
       ============================================================ */

    function executeCommand(
        command
    ) {
        const normalized =
            safeText(command)
                .trim()
                .toLowerCase();

        if (!normalized) {
            return;
        }

        closeCommandPalette();

        if (
            normalized.includes(
                "upload"
            )
        ) {
            triggerUpload();
            return;
        }

        if (
            normalized.includes(
                "analytic"
            )
        )
        {
            scrollToSection(
                "analytics"
            );
            return;
        }

        if (
            normalized.includes(
                "search"
            )
        ) {
            focusSearch();
            return;
        }

        if (
            normalized.includes(
                "memory"
            )
        ) {
            scrollToSection(
                "memories"
            );
            return;
        }

        if (
            normalized.includes(
                "dark"
            )
        ) {
            applyTheme("dark");
            return;
        }

        if (
            normalized.includes(
                "light"
            )
        ) {
            applyTheme("light");
        }
    }


    /* ============================================================
       SEARCH FOCUS
       ============================================================ */

    function getSearchInput() {
        return (
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
            )
        );
    }


    function focusSearch() {
        const input =
            getSearchInput();

        if (!input) {
            return;
        }

        input.scrollIntoView({
            behavior:
                prefersReducedMotion()
                    ? "auto"
                    : "smooth",
            block: "center",
        });

        focusElement(input);
    }


    /* ============================================================
       UPLOAD TRIGGER
       ============================================================ */

    function triggerUpload() {
        /*
         * If upload.js exists, use its public API.
         */
        if (
            MemoryOS.upload &&
            typeof MemoryOS.upload
                .open === "function"
        ) {
            MemoryOS.upload.open();
            return;
        }

        /*
         * Fallback to the file input.
         */
        const input =
            document.querySelector(
                "#fileInput"
            ) ||
            document.querySelector(
                "#uploadInput"
            ) ||
            document.querySelector(
                "input[type='file']"
            );

        if (input) {
            input.click();
        }

        window.dispatchEvent(
            new CustomEvent(
                "memoryos:open-upload"
            )
        );
    }


    /* ============================================================
       DRAWER
       ============================================================ */

    function openDrawer(
        data = null
    ) {
        if (
            !elements.drawer
        ) {
            return;
        }

        state.drawerOpen =
            true;

        elements.drawer.hidden =
            false;

        elements.drawer.classList.add(
            "is-open"
        );

        elements.drawer.setAttribute(
            "aria-hidden",
            "false"
        );

        updateBodyLock();

        if (data) {
            populateDrawer(
                data
            );
        }

        window.dispatchEvent(
            new CustomEvent(
                "memoryos:drawer-open",
                {
                    detail: data,
                }
            )
        );

        focusElement(
            elements.drawerClose
        );
    }


    function closeDrawer() {
        if (
            !elements.drawer
        ) {
            return;
        }

        state.drawerOpen =
            false;

        elements.drawer.classList.remove(
            "is-open"
        );

        elements.drawer.setAttribute(
            "aria-hidden",
            "true"
        );

        if (
            prefersReducedMotion()
        ) {
            elements.drawer.hidden =
                true;
        } else {
            window.setTimeout(
                () => {
                    if (
                        !state.drawerOpen
                    ) {
                        elements.drawer.hidden =
                            true;
                    }
                },
                300
            );
        }

        updateBodyLock();

        window.dispatchEvent(
            new CustomEvent(
                "memoryos:drawer-close"
            )
        );
    }


    /* ============================================================
       DRAWER DATA
       ============================================================ */

    function setText(
        selector,
        value
    ) {
        const element =
            elements.drawer?.querySelector(
                selector
            );

        if (!element) {
            return;
        }

        element.textContent =
            safeText(value);
    }


    function populateDrawer(
        data
    ) {
        if (
            !elements.drawer ||
            !data
        ) {
            return;
        }

        const memory =
            data.memory ||
            data.result ||
            data;

        setText(
            "[data-memory-title]",
            memory.title ||
                "Untitled memory"
        );

        setText(
            "[data-memory-summary]",
            memory.summary ||
                "No summary available."
        );

        const rawImage = firstDefined(
            memory.image_url,
            memory.preview_url,
            memory.thumbnail_url,
            memory.original_image_url,
            memory.image,
            memory.originalImage,
            memory.metadata?.image_url,
            memory.metadata?.preview_url,
            memory.metadata?.thumbnail_url,
            memory.metadata?.original_image_url,
            memory.file_url,
            memory.path
        );

        setText(
            "[data-memory-category]",
            memory.category ||
                "Uncategorized"
        );

        setText(
            "[data-memory-intent]",
            memory.intent ||
                "Unknown"
        );

        setText(
            "[data-memory-semantic-text]",
            memory.semantic_text ||
                memory.semanticText ||
                ""
        );

        setText(
            "[data-memory-ocr]",
            memory.ocr_text ||
                memory.ocrText ||
                ""
        );

        const image =
            elements.drawer.querySelector(
                "[data-memory-image]"
            );

        if (
            image &&
            rawImage
        ) {
            image.src =
                rawImage;

            image.alt =
                memory.title ||
                "Memory image";
            image.style.objectFit =
                "contain";
        }

        renderDrawerEntities(
            memory.entities
        );

        renderMatchExplanation(
            memory.explanation ||
                data.explanation
        );
    }


    function renderDrawerEntities(
        entities
    ) {
        const container =
            elements.drawer?.querySelector(
                "[data-memory-entities]"
            );

        if (!container) {
            return;
        }

        container.replaceChildren();

        if (
            !Array.isArray(
                entities
            ) ||
            entities.length === 0
        ) {
            const empty =
                document.createElement(
                    "span"
                );

            empty.className =
                "drawer-empty";

            empty.textContent =
                "No entities detected.";

            container.appendChild(
                empty
            );

            return;
        }

        const fragment =
            document.createDocumentFragment();

        entities.forEach(
            (entity) => {
                const tag =
                    document.createElement(
                        "span"
                    );

                tag.className =
                    "entity-chip";

                tag.textContent =
                    typeof entity ===
                    "string"
                        ? entity
                        : entity.name ||
                          entity.value ||
                          "Entity";

                fragment.appendChild(
                    tag
                );
            }
        );

        container.appendChild(
            fragment
        );
    }


    function renderMatchExplanation(
        explanation
    ) {
        const container =
            elements.drawer?.querySelector(
                "[data-match-explanation]"
            );

        if (!container) {
            return;
        }

        container.replaceChildren();

        if (
            !explanation
        ) {
            const empty =
                document.createElement(
                    "span"
                );

            empty.textContent =
                "Match explanation unavailable.";

            container.appendChild(
                empty
            );

            return;
        }

        if (
            Array.isArray(
                explanation
            )
        ) {
            explanation.forEach(
                (reason) => {
                    const item =
                        document.createElement(
                            "div"
                        );

                    item.className =
                        "match-reason";

                    item.textContent =
                        safeText(
                            reason
                        );

                    container.appendChild(
                        item
                    );
                }
            );

            return;
        }

        container.textContent =
            safeText(
                explanation
            );
    }


    /* ============================================================
       FILTER PANEL
       ============================================================ */

    function openFilterPanel() {
        if (
            !elements.filterPanel
        ) {
            return;
        }

        state.filterPanelOpen =
            true;

        elements.filterPanel.hidden =
            false;

        elements.filterPanel.classList.add(
            "is-open"
        );

        elements.filterPanel.setAttribute(
            "aria-hidden",
            "false"
        );

        updateBodyLock();

        focusElement(
            elements.filterClose
        );
    }


    function closeFilterPanel() {
        if (
            !elements.filterPanel
        ) {
            return;
        }

        state.filterPanelOpen =
            false;

        elements.filterPanel.classList.remove(
            "is-open"
        );

        elements.filterPanel.setAttribute(
            "aria-hidden",
            "true"
        );

        if (
            prefersReducedMotion()
        ) {
            elements.filterPanel.hidden =
                true;
        } else {
            window.setTimeout(
                () => {
                    if (
                        !state.filterPanelOpen
                    ) {
                        elements.filterPanel.hidden =
                            true;
                    }
                },
                240
            );
        }

        updateBodyLock();
    }


    function toggleFilterPanel() {
        if (
            state.filterPanelOpen
        ) {
            closeFilterPanel();
        } else {
            openFilterPanel();
        }
    }


    /* ============================================================
       TOAST SYSTEM
       ============================================================ */

    function ensureToastContainer() {
        if (
            elements.toastContainer
        ) {
            return elements.toastContainer;
        }

        const container =
            document.createElement(
                "div"
            );

        container.id =
            "toastContainer";

        container.className =
            "toast-container";

        container.setAttribute(
            "aria-live",
            "polite"
        );

        container.setAttribute(
            "aria-atomic",
            "true"
        );

        elements.body.appendChild(
            container
        );

        elements.toastContainer =
            container;

        return container;
    }


    function toast(
        message,
        type = "info",
        duration = 4200
    ) {
        const container =
            ensureToastContainer();

        const id =
            ++state.toastCounter;

        const toastElement =
            document.createElement(
                "div"
            );

        toastElement.className =
            `toast toast--${type}`;

        toastElement.dataset.toastId =
            String(id);

        toastElement.setAttribute(
            "role",
            type === "error"
                ? "alert"
                : "status"
        );

        const content =
            document.createElement(
                "div"
            );

        content.className =
            "toast__content";

        const text =
            document.createElement(
                "div"
            );

        text.className =
            "toast__message";

        text.textContent =
            safeText(message);

        const close =
            document.createElement(
                "button"
            );

        close.type = "button";

        close.className =
            "toast__close";

        close.setAttribute(
            "aria-label",
            "Dismiss notification"
        );

        close.textContent =
            "×";

        close.addEventListener(
            "click",
            () => {
                dismissToast(
                    toastElement
                );
            }
        );

        content.appendChild(
            text
        );

        toastElement.appendChild(
            content
        );

        toastElement.appendChild(
            close
        );

        container.appendChild(
            toastElement
        );

        window.requestAnimationFrame(
            () => {
                toastElement.classList.add(
                    "is-visible"
                );
            }
        );

        if (
            duration > 0
        ) {
            window.setTimeout(
                () => {
                    dismissToast(
                        toastElement
                    );
                },
                duration
            );
        }

        return id;
    }


    function dismissToast(
        toastElement
    ) {
        if (
            !toastElement ||
            !toastElement.isConnected
        ) {
            return;
        }

        toastElement.classList.remove(
            "is-visible"
        );

        if (
            prefersReducedMotion()
        ) {
            toastElement.remove();
            return;
        }

        window.setTimeout(
            () => {
                toastElement.remove();
            },
            260
        );
    }


    /* ============================================================
       SCROLL NAVIGATION
       ============================================================ */

    function scrollToSection(
        sectionId
    ) {
        const section =
            document.getElementById(
                sectionId
            );

        if (!section) {
            return;
        }

        closeMobileNavigation();

        section.scrollIntoView({
            behavior:
                prefersReducedMotion()
                    ? "auto"
                    : "smooth",
            block: "start",
        });

        setActiveSection(
            sectionId
        );
    }


    function setActiveSection(
        sectionId
    ) {
        state.activeSection =
            sectionId;

        elements.navLinks.forEach(
            (link) => {
                const href =
                    link.getAttribute(
                        "href"
                    );

                const target =
                    href?.startsWith(
                        "#"
                    )
                        ? href.slice(1)
                        : link.dataset
                              .section;

                const active =
                    target ===
                    sectionId;

                link.classList.toggle(
                    "is-active",
                    active
                );

                if (active) {
                    link.setAttribute(
                        "aria-current",
                        "page"
                    );
                } else {
                    link.removeAttribute(
                        "aria-current"
                    );
                }
            }
        );
    }


    /* ============================================================
       INTERSECTION OBSERVER
       ============================================================ */

    function initializeSectionObserver() {
        if (
            !("IntersectionObserver" in
                window)
        ) {
            return;
        }

        if (
            elements.sections.length ===
            0
        ) {
            return;
        }

        const observer =
            new IntersectionObserver(
                (
                    entries
                ) => {
                    const visible =
                        entries
                            .filter(
                                (
                                    entry
                                ) =>
                                    entry.isIntersecting
                            )
                            .sort(
                                (
                                    a,
                                    b
                                ) =>
                                    b.intersectionRatio -
                                    a.intersectionRatio
                            );

                    if (
                        visible.length
                    ) {
                        setActiveSection(
                            visible[0]
                                .target
                                .id
                        );
                    }
                },
                {
                    threshold: [
                        0.2,
                        0.4,
                        0.6,
                    ],
                    rootMargin:
                        "-12% 0px -45% 0px",
                }
            );

        elements.sections.forEach(
            (section) =>
                observer.observe(
                    section
                )
        );
    }


    /* ============================================================
       HEADER SCROLL EFFECT
       ============================================================ */

    function initializeHeaderScroll() {
        if (
            !elements.header
        ) {
            return;
        }

        let ticking = false;

        function update() {
            const scrolled =
                window.scrollY >
                24;

            elements.header.classList.toggle(
                "is-scrolled",
                scrolled
            );

            if (
                elements.scrollTop
            ) {
                elements.scrollTop.classList.toggle(
                    "is-visible",
                    window.scrollY >
                        600
                );
            }

            ticking = false;
        }

        window.addEventListener(
            "scroll",
            () => {
                if (!ticking) {
                    window.requestAnimationFrame(
                        update
                    );

                    ticking = true;
                }
            },
            {
                passive: true,
            }
        );

        update();
    }


    /* ============================================================
       NAVIGATION EVENTS
       ============================================================ */

    function bindNavigation() {
        elements.navLinks.forEach(
            (link) => {
                link.addEventListener(
                    "click",
                    (event) => {
                        const href =
                            link.getAttribute(
                                "href"
                            );

                        if (
                            !href ||
                            !href.startsWith(
                                "#"
                            )
                        ) {
                            return;
                        }

                        const id =
                            href.slice(1);

                        const target =
                            document.getElementById(
                                id
                            );

                        if (!target) {
                            return;
                        }

                        event.preventDefault();

                        scrollToSection(
                            id
                        );
                    }
                );
            }
        );
    }


    /* ============================================================
       CLICK ACTIONS
       ============================================================ */

    function bindGlobalActions() {
        document.addEventListener(
            "click",
            (event) => {
                const actionElement =
                    event.target.closest(
                        "[data-ui-action]"
                    );

                if (!actionElement) {
                    return;
                }

                const action =
                    actionElement.dataset
                        .uiAction;

                switch (action) {
                    case "upload":
                        event.preventDefault();
                        triggerUpload();
                        break;

                    case "search":
                        event.preventDefault();
                        focusSearch();
                        break;

                    case "analytics":
                        event.preventDefault();
                        scrollToSection(
                            "analytics"
                        );
                        break;

                    case "memories":
                        event.preventDefault();
                        scrollToSection(
                            "memories"
                        );
                        break;

                    case "command":
                        event.preventDefault();
                        openCommandPalette();
                        break;

                    case "filters":
                        event.preventDefault();
                        toggleFilterPanel();
                        break;

                    case "theme":
                        event.preventDefault();
                        toggleTheme();
                        break;

                    case "close-drawer":
                        event.preventDefault();
                        closeDrawer();
                        break;

                    case "close-command":
                        event.preventDefault();
                        closeCommandPalette();
                        break;

                    case "close-filters":
                        event.preventDefault();
                        closeFilterPanel();
                        break;
                }
            }
        );
    }


    /* ============================================================
       BACKDROP EVENTS
       ============================================================ */

    function bindOverlayEvents() {
        elements.commandBackdrop?.addEventListener(
            "click",
            closeCommandPalette
        );

        elements.commandClose?.addEventListener(
            "click",
            closeCommandPalette
        );

        elements.drawerBackdrop?.addEventListener(
            "click",
            closeDrawer
        );

        elements.drawerClose?.addEventListener(
            "click",
            closeDrawer
        );

        elements.filterBackdrop?.addEventListener(
            "click",
            closeFilterPanel
        );

        elements.filterClose?.addEventListener(
            "click",
            closeFilterPanel
        );

        elements.mobileClose?.addEventListener(
            "click",
            closeMobileNavigation
        );

        elements.mobileToggle?.addEventListener(
            "click",
            toggleMobileNavigation
        );
    }


    /* ============================================================
       THEME EVENTS
       ============================================================ */

    function bindTheme() {
        elements.themeToggle?.addEventListener(
            "click",
            toggleTheme
        );

        applyTheme(
            state.theme
        );
    }


    /* ============================================================
       COMMAND PALETTE EVENTS
       ============================================================ */

    function bindCommandPalette() {
        elements.commandInput?.addEventListener(
            "keydown",
            (event) => {
                if (
                    event.key ===
                    "Enter"
                ) {
                    executeCommand(
                        event.target
                            .value
                    );

                    event.target.value =
                        "";
                }
            }
        );

        elements.commandPalette?.addEventListener(
            "click",
            (event) => {
                const command =
                    event.target.closest(
                        "[data-command]"
                    );

                if (!command) {
                    return;
                }

                const value =
                    command.dataset
                        .command;

                executeCommand(
                    value
                );
            }
        );
    }


    /* ============================================================
       KEYBOARD SHORTCUTS
       ============================================================ */

    function isTypingTarget(
        target
    ) {
        if (!target) {
            return false;
        }

        const tag =
            target.tagName?.toLowerCase();

        return (
            tag === "input" ||
            tag === "textarea" ||
            tag === "select" ||
            target.isContentEditable
        );
    }


    function bindKeyboardShortcuts() {
        document.addEventListener(
            "keydown",
            (event) => {
                const modifier =
                    event.ctrlKey ||
                    event.metaKey;

                /*
                 * Command palette
                 *
                 * Ctrl + K
                 * Cmd + K
                 */
                if (
                    modifier &&
                    event.key.toLowerCase() ===
                        "k"
                ) {
                    event.preventDefault();

                    toggleCommandPalette();

                    return;
                }

                /*
                 * Search
                 *
                 * /
                 */
                if (
                    event.key === "/" &&
                    !isTypingTarget(
                        event.target
                    )
                ) {
                    event.preventDefault();

                    focusSearch();

                    return;
                }

                /*
                 * Escape
                 */
                if (
                    event.key ===
                    "Escape"
                ) {
                    if (
                        state.commandPaletteOpen
                    ) {
                        closeCommandPalette();
                        return;
                    }

                    if (
                        state.drawerOpen
                    ) {
                        closeDrawer();
                        return;
                    }

                    if (
                        state.filterPanelOpen
                    ) {
                        closeFilterPanel();
                        return;
                    }

                    if (
                        state.mobileNavigationOpen
                    ) {
                        closeMobileNavigation();
                    }
                }
            }
        );
    }


    /* ============================================================
       SCROLL TOP
       ============================================================ */

    function bindScrollTop() {
        elements.scrollTop?.addEventListener(
            "click",
            () => {
                window.scrollTo({
                    top: 0,
                    behavior:
                        prefersReducedMotion()
                            ? "auto"
                            : "smooth",
                });
            }
        );
    }


    /* ============================================================
       MEMORY CARD EVENTS
       ============================================================ */

    function bindMemoryEvents() {
        document.addEventListener(
            "click",
            (event) => {
                const card =
                    event.target.closest(
                        "[data-memory-id]"
                    );

                if (!card) {
                    return;
                }

                /*
                 * Don't interfere with buttons
                 * inside a memory card.
                 */
                if (
                    event.target.closest(
                        "button, a, input"
                    )
                ) {
                    return;
                }

                const memoryId =
                    card.dataset.memoryId;

                if (!memoryId) {
                    return;
                }

                window.dispatchEvent(
                    new CustomEvent(
                        "memoryos:open-memory",
                        {
                            detail: {
                                memoryId,
                            },
                        }
                    )
                );
            }
        );
    }


    /* ============================================================
       UPLOAD / SEARCH / RESULT / STATS BRIDGE
       ============================================================ */

    function bindModuleEvents() {
        /*
         * Upload completed
         */
        window.addEventListener(
            "memoryos:upload-complete",
            () => {
                /*
                 * Tell stats/results modules
                 * to refresh themselves.
                 */
                window.dispatchEvent(
                    new CustomEvent(
                        "memoryos:refresh"
                    )
                );
            }
        );


        /*
         * Memory selected from results
         */
        window.addEventListener(
            "memoryos:memory-selected",
            (event) => {
                openDrawer(
                    event.detail
                );
            }
        );


        /*
         * Results can explicitly
         * request the drawer.
         */
        window.addEventListener(
            "memoryos:open-drawer",
            (event) => {
                openDrawer(
                    event.detail
                );
            }
        );


        /*
         * Global toast bridge.
         */
        window.addEventListener(
            "memoryos:toast",
            (event) => {
                const detail =
                    event.detail || {};

                toast(
                    detail.message ||
                        "MemoryOS notification",
                    detail.type ||
                        "info",
                    detail.duration ??
                        4200
                );
            }
        );


        /*
         * Search loading.
         */
        window.addEventListener(
            "memoryos:search-loading",
            () => {
                elements.body.classList.add(
                    "is-searching"
                );
            }
        );


        /*
         * Search completed.
         */
        window.addEventListener(
            "memoryos:search-complete",
            () => {
                elements.body.classList.remove(
                    "is-searching"
                );
            }
        );


        /*
         * Search error.
         */
        window.addEventListener(
            "memoryos:search-error",
            () => {
                elements.body.classList.remove(
                    "is-searching"
                );
            }
        );
    }


    /* ============================================================
       ACCESSIBILITY
       ============================================================ */

    function initializeAccessibility() {
        /*
         * Prevent focus from being visually lost
         * when UI overlays are open.
         */
        document.addEventListener(
            "focusin",
            (event) => {
                const overlayOpen =
                    state.commandPaletteOpen ||
                    state.drawerOpen ||
                    state.filterPanelOpen;

                if (!overlayOpen) {
                    return;
                }

                /*
                 * We intentionally don't forcibly
                 * trap focus here because each overlay
                 * may have its own internal controls.
                 */
            }
        );


        /*
         * Add keyboard-friendly class.
         */
        document.addEventListener(
            "keydown",
            (event) => {
                if (
                    event.key ===
                    "Tab"
                ) {
                    elements.body.classList.add(
                        "using-keyboard"
                    );
                }
            }
        );

        document.addEventListener(
            "mousedown",
            () => {
                elements.body.classList.remove(
                    "using-keyboard"
                );
            }
        );
    }


    /* ============================================================
       ONLINE / OFFLINE STATE
       ============================================================ */

    function initializeNetworkState() {
        function updateNetworkState() {
            const online =
                navigator.onLine;

            elements.body.classList.toggle(
                "is-offline",
                !online
            );

            window.dispatchEvent(
                new CustomEvent(
                    "memoryos:network-change",
                    {
                        detail: {
                            online,
                        },
                    }
                )
            );

            if (!online) {
                toast(
                    "MemoryOS is offline. Some features may be unavailable.",
                    "warning",
                    6000
                );
            }
        }

        window.addEventListener(
            "online",
            updateNetworkState
        );

        window.addEventListener(
            "offline",
            updateNetworkState
        );

        updateNetworkState();
    }


    /* ============================================================
       PREMIUM MICRO-INTERACTION
       ============================================================ */

    function initializeCardTilt() {
        /*
         * Keep this extremely subtle.
         * No aggressive 3D movement.
         */
        if (
            prefersReducedMotion()
        ) {
            return;
        }

        if (
            window.matchMedia(
                "(pointer: coarse)"
            ).matches
        ) {
            return;
        }

        const cards =
            document.querySelectorAll(
                "[data-premium-tilt]"
            );

        cards.forEach(
            (card) => {
                card.addEventListener(
                    "pointermove",
                    (event) => {
                        const rect =
                            card.getBoundingClientRect();

                        const x =
                            event.clientX -
                            rect.left;

                        const y =
                            event.clientY -
                            rect.top;

                        const rotateX =
                            ((y /
                                rect.height) -
                                0.5) *
                            -2;

                        const rotateY =
                            ((x /
                                rect.width) -
                                0.5) *
                            2;

                        card.style.setProperty(
                            "--tilt-x",
                            `${rotateX}deg`
                        );

                        card.style.setProperty(
                            "--tilt-y",
                            `${rotateY}deg`
                        );
                    }
                );

                card.addEventListener(
                    "pointerleave",
                    () => {
                        card.style.setProperty(
                            "--tilt-x",
                            "0deg"
                        );

                        card.style.setProperty(
                            "--tilt-y",
                            "0deg"
                        );
                    }
                );
            }
        );
    }


    /* ============================================================
       INITIAL UI STATE
       ============================================================ */

    function initializeUiState() {
        if (
            elements.commandPalette
        ) {
            elements.commandPalette.hidden =
                true;

            elements.commandPalette.setAttribute(
                "aria-hidden",
                "true"
            );
        }

        if (
            elements.drawer
        ) {
            elements.drawer.hidden =
                true;

            elements.drawer.setAttribute(
                "aria-hidden",
                "true"
            );
        }

        if (
            elements.filterPanel
        ) {
            elements.filterPanel.hidden =
                true;

            elements.filterPanel.setAttribute(
                "aria-hidden",
                "true"
            );
        }

        if (
            elements.mobileMenu
        ) {
            elements.mobileMenu.hidden =
                true;

            elements.mobileMenu.setAttribute(
                "aria-hidden",
                "true"
            );
        }
    }


    /* ============================================================
       INITIALIZATION
       ============================================================ */

    function initialize() {
        if (
            state.initialized
        ) {
            return;
        }

        state.initialized =
            true;

        initializeUiState();

        bindTheme();

        bindNavigation();

        bindGlobalActions();

        bindOverlayEvents();

        bindCommandPalette();

        bindKeyboardShortcuts();

        bindScrollTop();

        bindMemoryEvents();

        bindModuleEvents();

        initializeAccessibility();

        initializeNetworkState();

        initializeSectionObserver();

        initializeHeaderScroll();

        initializeCardTilt();

        /*
         * Default section.
         */
        setActiveSection(
            state.activeSection
        );

        /*
         * Expose ready state.
         */
        document.documentElement.classList.add(
            "memoryos-ready"
        );

        window.dispatchEvent(
            new CustomEvent(
                "memoryos:app-ready"
            )
        );

        /*
         * Development logging only.
         */
        if (
            MemoryOS.config?.debug
                ?.enabled
        ) {
            console.info(
                "[MemoryOS] Premium UI initialized."
            );
        }
    }


    /* ============================================================
       PUBLIC API
       ============================================================ */

    MemoryOS.ui = Object.freeze({
        openDrawer,

        closeDrawer,

        openCommandPalette,

        closeCommandPalette,

        toggleCommandPalette,

        openFilterPanel,

        closeFilterPanel,

        toggleFilterPanel,

        openMobileNavigation,

        closeMobileNavigation,

        toggleMobileNavigation,

        focusSearch,

        triggerUpload,

        scrollToSection,

        applyTheme,

        toggleTheme,

        toast,

        getState() {
            return {
                ...state,
            };
        },
    });


    /*
     * Also expose toast directly for other modules.
     */
    MemoryOS.toast =
        toast;


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