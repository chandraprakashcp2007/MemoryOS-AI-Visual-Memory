import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  Brain,
  Check,
  ChevronRight,
  Clock3,
  Database,
  FileImage,
  FileText,
  Image as ImageIcon,
  Layers3,
  Loader2,
  Menu,
  Search,
  Sun,
  Moon,
  Monitor,
  Copy,
  Sparkles,
  Trash2,
  Upload,
  X,
  Zap,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";

import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState("search");

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const [memories, setMemories] = useState([]);
  const [memoryTotal, setMemoryTotal] = useState(0);
  const [memoryPage, setMemoryPage] = useState(1);
  const [hasMoreMemories, setHasMoreMemories] = useState(false);
  const [memoryCategory, setMemoryCategory] = useState("");
  const [memorySort, setMemorySort] = useState("newest");
  const [stats, setStats] = useState(null);

  const [loading, setLoading] = useState(false);
  const [dashboardLoading, setDashboardLoading] = useState(true);

  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");

  const [uploadOpen, setUploadOpen] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem("memoryos-sidebar") === "collapsed");

  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [predictiveSuggestions, setPredictiveSuggestions] = useState([]);
  const [recentSearches, setRecentSearches] = useState(() => {
    try { return JSON.parse(localStorage.getItem("memoryos-recent-searches") || "[]"); }
    catch { return []; }
  });
  const [themePreference, setThemePreference] = useState(() => localStorage.getItem("memoryos-theme") || "dark");

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const applyTheme = () => {
      const resolved = themePreference === "system" ? (media.matches ? "light" : "dark") : themePreference;
      document.documentElement.dataset.theme = resolved;
    };
    applyTheme();
    media.addEventListener("change", applyTheme);
    return () => media.removeEventListener("change", applyTheme);
  }, [themePreference]);

  function changeTheme(value) {
    localStorage.setItem("memoryos-theme", value);
    setThemePreference(value);
  }

  function toggleSidebar() {
    setSidebarCollapsed((current) => {
      const next = !current;
      localStorage.setItem("memoryos-sidebar", next ? "collapsed" : "expanded");
      return next;
    });
  }

  const fileInputRef = useRef(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard({ category = memoryCategory, sort = memorySort } = {}) {
    setDashboardLoading(true);

    try {
      const [memoryResponse, statsResponse] = await Promise.all([
        fetch(`${API}/memories?limit=100&sort=${encodeURIComponent(sort)}${category ? `&category=${encodeURIComponent(category)}` : ""}`),
        fetch(`${API}/stats`),
      ]);

      if (memoryResponse.ok) {
        const data = await memoryResponse.json();

        const loaded =
          data?.memories ||
          data?.results ||
          data?.items ||
          [];

        setMemories(Array.isArray(loaded) ? loaded : []);
        setMemoryTotal(Number(data?.total) || (Array.isArray(loaded) ? loaded.length : 0));
        setMemoryPage(Number(data?.page) || 1);
        setHasMoreMemories(Boolean(data?.has_next));
      } else {
        console.error(
          "Memories request failed:",
          memoryResponse.status
        );
      }

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error("Dashboard error:", err);
    } finally {
      setDashboardLoading(false);
    }
  }

  async function loadMoreMemories() {
    if (dashboardLoading || !hasMoreMemories) return;

    try {
      const response = await fetch(`${API}/memories?page=${memoryPage + 1}&limit=100&sort=${encodeURIComponent(memorySort)}${memoryCategory ? `&category=${encodeURIComponent(memoryCategory)}` : ""}`);
      if (!response.ok) throw new Error(`Unable to load memories (${response.status})`);
      const data = await response.json();
      const next = Array.isArray(data?.memories) ? data.memories : [];
      setMemories((current) => {
        const merged = [...current, ...next];
        return merged.filter((memory, index) => {
          const id = memory?.memory_id ?? memory?.id;
          return index === merged.findIndex((item) => (item?.memory_id ?? item?.id) === id);
        });
      });
      setMemoryPage(Number(data?.page) || memoryPage + 1);
      setMemoryTotal(Number(data?.total) || memoryTotal);
      setHasMoreMemories(Boolean(data?.has_next));
    } catch (err) {
      console.error("Load more memories error:", err);
    }
  }

  function updateLibraryFilter(category, sort = memorySort) {
    setMemoryCategory(category);
    setMemorySort(sort);
    setMemoryPage(1);
    loadDashboard({ category, sort });
  }

  async function handleSearch(event, customQuery = null) {
    event?.preventDefault();

    const searchText =
      typeof customQuery === "string"
        ? customQuery.trim()
        : query.trim();

    if (!searchText) return;

    setQuery(searchText);
    setLoading(true);
    setSearched(true);
    setError("");
    setPredictiveSuggestions([]);

    try {
      const normalized = searchText.toLowerCase();

      if (
        normalized === "everything" ||
        normalized === "all" ||
        normalized === "all screenshots"
      ) {
        const response = await fetch(
          `${API}/memories?limit=100`
        );

        if (!response.ok) {
          throw new Error(
            `Unable to load memories (${response.status})`
          );
        }

        const data = await response.json();

        const all =
          data?.memories ||
          data?.results ||
          data?.items ||
          [];

        setResults(
          Array.isArray(all) ? all : []
        );

        return;
      }

      const response = await fetch(`${API}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: searchText,
          limit: 20,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Search failed (${response.status})`
        );
      }

      const data = await response.json();

      const found =
        data?.results ||
        data?.memories ||
        data?.items ||
        [];

      setResults(
        Array.isArray(found) ? found : []
      );
      setRecentSearches((current) => {
        const next = [searchText, ...current.filter((item) => item.toLowerCase() !== searchText.toLowerCase())].slice(0, 6);
        localStorage.setItem("memoryos-recent-searches", JSON.stringify(next));
        return next;
      });
    } catch (err) {
      console.error("Search error:", err);

      setError(
        err?.message ||
          "Unable to search your memories."
      );

      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  async function updateQuery(value) {
    setQuery(value);
    const trimmed = value.trim();
    if (trimmed.length < 2) {
      setPredictiveSuggestions([]);
      return;
    }
    try {
      const response = await fetch(`${API}/search/suggestions?q=${encodeURIComponent(trimmed)}&limit=6`);
      const data = response.ok ? await response.json() : {};
      setPredictiveSuggestions(Array.isArray(data?.suggestions) ? data.suggestions : []);
    } catch {
      setPredictiveSuggestions([]);
    }
  }

  async function sendFeedback(memoryId, relevant) {
    if (!query || !memoryId) return;
    try {
      await fetch(`${API}/search/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, memory_id: memoryId, relevant }),
      });
    } catch (feedbackError) {
      console.error("Search feedback failed", feedbackError);
    }
  }

  function clearSearch() {
    setQuery("");
    setResults([]);
    setSearched(false);
    setError("");
  }

  function suggestedSearch(value) {
    handleSearch(null, value);
  }

  /*
   * ============================================================
   * IMAGE URL
   * ============================================================
   *
   * The backend endpoint is:
   *
   * /upload/file/{memory_id}
   *
   * IMPORTANT:
   * Always prefer memory_id.
   */

  function imageUrl(memory, { thumbnail = false } = {}) {
    if (!memory) return null;

    const suppliedUrl = thumbnail
      ? memory.thumbnail_url ?? memory.thumbnailUrl
      : memory.original_image_url ??
        memory.originalImageUrl ??
        memory.image_url ??
        memory.imageUrl;

    const imageVersion = encodeURIComponent(
      String(memory.image_hash ?? memory.imageHash ?? memory.sha256 ?? memory.file_hash ?? memory.memory_id ?? memory.memoryId ?? memory.id ?? "")
    );

    const withVersion = (url) =>
      imageVersion
        ? `${url}${url.includes("?") ? "&" : "?"}v=${imageVersion}`
        : url;

    if (typeof suppliedUrl === "string" && suppliedUrl.trim()) {
      const normalizedUrl = suppliedUrl.trim();

      if (/^https?:\/\//i.test(normalizedUrl)) {
        return withVersion(normalizedUrl);
      }

      if (normalizedUrl.startsWith("/")) {
        return withVersion(`${API}${normalizedUrl}`);
      }
    }

    const memoryId =
      memory.memory_id ??
      memory.memoryId ??
      memory.id;

    if (
      memoryId === undefined ||
      memoryId === null ||
      String(memoryId).trim() === ""
    ) {
      return null;
    }

    const baseUrl = `${API}/upload/file/${encodeURIComponent(
      String(memoryId)
    )}`;

    return withVersion(thumbnail ? `${baseUrl}?thumbnail=true` : baseUrl);
  }

  function openUpload() {
    setUploadOpen(true);
    setMobileMenu(false);
  }

  function closeUpload() {
    if (uploading) return;

    setUploadOpen(false);
    setUploadFiles([]);
    setUploadStatus("");
  }

  function addFiles(files) {
    const valid = Array.from(files || []).filter(
      (file) => {
        const validType = [
          "image/png",
          "image/jpeg",
          "image/webp",
        ].includes(file.type);

        const validSize =
          file.size <= 10 * 1024 * 1024;

        return validType && validSize;
      }
    );

    setUploadFiles((current) => {
      const combined = [
        ...current,
        ...valid,
      ];

      const unique = combined.filter(
        (file, index, array) =>
          index ===
          array.findIndex(
            (item) =>
              item.name === file.name &&
              item.size === file.size &&
              item.lastModified ===
                file.lastModified
          )
      );

      return unique.slice(0, 50);
    });
  }

  function removeUploadFile(index) {
    setUploadFiles((files) =>
      files.filter((_, i) => i !== index)
    );
  }

  async function uploadScreenshots() {
    if (!uploadFiles.length || uploading) {
      return;
    }

    setUploading(true);

    setUploadStatus(
      `Uploading ${uploadFiles.length} screenshot${
        uploadFiles.length === 1 ? "" : "s"
      }...`
    );

    try {
      let successful = 0;
      let failed = 0;

      for (
        let i = 0;
        i < uploadFiles.length;
        i++
      ) {
        const file = uploadFiles[i];

        setUploadStatus(
          `Processing ${i + 1} of ${
            uploadFiles.length
          }: ${file.name}`
        );

        const formData = new FormData();

        formData.append("file", file);

        try {
          const response = await fetch(
            `${API}/upload`,
            {
              method: "POST",
              body: formData,
            }
          );

          if (response.ok) {
            successful++;
          } else {
            failed++;

            console.error(
              "Upload failed:",
              file.name,
              response.status
            );
          }
        } catch (uploadError) {
          failed++;

          console.error(
            "Upload request failed:",
            file.name,
            uploadError
          );
        }
      }

      if (failed === 0) {
        setUploadStatus(
          `✓ ${successful} screenshot${
            successful === 1 ? "" : "s"
          } uploaded successfully`
        );
      } else {
        setUploadStatus(
          `✓ ${successful} uploaded · ${failed} failed`
        );
      }

      await loadDashboard();

      setTimeout(() => {
        setUploadFiles([]);
        setUploadOpen(false);
        setUploadStatus("");
      }, 1500);
    } catch (err) {
      console.error(
        "Upload error:",
        err
      );

      setUploadStatus(
        "Upload failed. Check that the MemoryOS backend is running."
      );
    } finally {
      setUploading(false);
    }
  }

  const memoryCount =
    Number.isFinite(memoryTotal) && memoryTotal > 0
      ? memoryTotal
      : Number(stats?.memories?.total_memories ?? stats?.total_memories ?? stats?.total ?? 0) || memories.length;

  const imageCount =
    Number(stats?.memories?.indexed_memories ?? stats?.total_images ?? stats?.images ?? 0) || memories.length;

  const suggestions = useMemo(() => {
    const concepts = new Set();

    memories.forEach((memory) => {
      if (memory?.category) {
        concepts.add(
          String(memory.category)
        );
      }
      (memory?.keywords || []).slice(0, 5).forEach((keyword) => concepts.add(String(keyword)));
    });

    return Array.from(
      concepts
    ).slice(0, 8);
  }, [memories]);

  function navigate(page) {
    setActivePage(page);
    setMobileMenu(false);

    if (page === "upload") {
      openUpload();
      return;
    }

    if (page === "search") {
      clearSearch();
      return;
    }

    if (page === "memories") {
      handleSearch(null, "everything");
      return;
    }
  }

  return (
    <div className="app">

      {/* =====================================================
          MOBILE HEADER
      ===================================================== */}

      <header className="mobile-header">
        <div className="mobile-brand">
          <div className="brand-logo">
            <Brain size={19} />
          </div>

          <span>MemoryOS</span>
        </div>

        <button
          type="button"
          className="icon-button"
          onClick={() =>
            setMobileMenu(
              (value) => !value
            )
          }
        >
          {mobileMenu ? (
            <X size={21} />
          ) : (
            <Menu size={21} />
          )}
        </button>
      </header>

      {/* =====================================================
          SIDEBAR
      ===================================================== */}

      <aside
        className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${
          mobileMenu
            ? "sidebar-open"
            : ""
        }`}
      >
        <div className="sidebar-top">

          <div className="brand">
            <div className="brand-logo">
              <Brain size={20} />
            </div>

            <div className="brand-copy">
              <strong>
                MemoryOS
              </strong>

              <span>
                Visual memory engine
              </span>
            </div>

            <button type="button" className="sidebar-toggle" onClick={toggleSidebar} aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
              {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
            </button>
          </div>

          <div className="workspace-card">
            <div className="workspace-icon">
              <Sparkles size={15} />
            </div>

            <div>
              <span>
                Workspace
              </span>

              <strong>
                Personal Memory
              </strong>
            </div>

            <ChevronRight size={15} />
          </div>

          <nav className="nav">

            <NavItem
              icon={
                <Search size={17} />
              }
              label="Search"
              active={
                activePage ===
                "search"
              }
              onClick={() =>
                navigate("search")
              }
            />

            <NavItem
              icon={
                <Layers3 size={17} />
              }
              label="Memories"
              active={
                activePage ===
                "memories"
              }
              badge={memoryCount}
              onClick={() =>
                navigate("memories")
              }
            />

            <NavItem
              icon={
                <Upload size={17} />
              }
              label="Upload"
              active={
                activePage ===
                "upload"
              }
              onClick={() =>
                navigate("upload")
              }
            />

            <NavItem
              icon={
                <BarChart3 size={17} />
              }
              label="Analytics"
              active={
                activePage ===
                "analytics"
              }
              onClick={() => {
                setActivePage(
                  "analytics"
                );
                setMobileMenu(false);
              }}
            />
          </nav>
        </div>

        <div className="sidebar-bottom">

          <div className="engine-card">
            <div className="engine-indicator">
              <span />
            </div>

            <div className="engine-copy">
              <strong>
                Memory engine
              </strong>

              <small>
                Local memory index
              </small>
            </div>

            <div className="engine-status-dot">
              <Check size={10} />
            </div>
          </div>

          <div className="creator-card">

            <div className="creator-avatar">
              <span>CP</span>
            </div>

            <div className="creator-info">
              <span className="creator-label">
                CREATED BY
              </span>

              <strong>
                Chandra Prakash
              </strong>

              <small>
                AI Visual Memory Engine
              </small>
            </div>

            <div className="creator-spark">
              <Sparkles size={13} />
            </div>
          </div>

          <div className="sidebar-footer">
            <span>
              <Brain size={12} />
              MemoryOS
            </span>

            <span>
              v1.0 MVP
            </span>
          </div>
        </div>
      </aside>

      {/* =====================================================
          MAIN
      ===================================================== */}

      <main className="main">

        <header className="topbar">

          <div className="breadcrumb">
            <span>
              MemoryOS
            </span>

            <ChevronRight size={14} />

            <strong>
              {activePage ===
              "analytics"
                ? "Analytics"
                : activePage ===
                  "memories"
                ? "Memories"
                : "Semantic Search"}
            </strong>
          </div>

          <div className="topbar-actions">

            <ThemeSwitcher value={themePreference} onChange={changeTheme} />

            <div className="system-status" title="Status reflects the local memory engine; visual analysis may be degraded when Gemini is unavailable.">
              <span className="pulse" />

              <span>
                Local memory index
              </span>
            </div>

            <button
              type="button"
              className="top-upload"
              onClick={openUpload}
            >
              <Upload size={16} />
              Upload
            </button>
          </div>
        </header>

        {/* SEARCH */}

        {activePage ===
          "search" && (
          <SearchPage
            query={query}
            setQuery={updateQuery}
            results={results}
            loading={loading}
            searched={searched}
            error={error}
            memories={memories}
            memoryCount={memoryCount}
            imageUrl={imageUrl}
            handleSearch={handleSearch}
            clearSearch={clearSearch}
            suggestedSearch={
              suggestedSearch
            }
            suggestions={suggestions}
            predictiveSuggestions={predictiveSuggestions}
            recentSearches={recentSearches}
            sendFeedback={sendFeedback}
            dashboardLoading={
              dashboardLoading
            }
          />
        )}

        {/* MEMORIES */}

        {activePage ===
          "memories" && (
          <MemoriesPage
            memories={
              results.length
                ? results
                : memories
            }
            memoryCount={
              memoryCount
            }
            hasMoreMemories={hasMoreMemories}
            loadMoreMemories={loadMoreMemories}
            categories={Array.from(new Set(memories.map((memory) => memory?.category).filter(Boolean))).sort()}
            activeCategory={memoryCategory}
            activeSort={memorySort}
            updateLibraryFilter={updateLibraryFilter}
            imageUrl={imageUrl}
            loading={loading}
          />
        )}

        {/* ANALYTICS */}

        {activePage ===
          "analytics" && (
          <AnalyticsPage
            memoryCount={
              memoryCount
            }
            imageCount={
              imageCount
            }
            memories={memories}
          />
        )}

        <footer className="footer">
          <div className="footer-brand">
            <Brain size={15} />
            <span>
              MemoryOS
            </span>
          </div>

          <span>
            Screenshots → OCR → AI →
            Embeddings → Semantic Search
          </span>

          <span className="footer-live">
            <span />
            Local & private
          </span>
        </footer>
      </main>

      {/* UPLOAD MODAL */}

      {uploadOpen && (
        <UploadModal
          files={uploadFiles}
          uploading={uploading}
          status={uploadStatus}
          fileInputRef={
            fileInputRef
          }
          addFiles={addFiles}
          removeFile={
            removeUploadFile
          }
          upload={
            uploadScreenshots
          }
          close={closeUpload}
        />
      )}
    </div>
  );
}


/* ============================================================
   NAV ITEM
============================================================ */

function NavItem({
  icon,
  label,
  active,
  badge,
  onClick,
}) {
  return (
    <button
      type="button"
      className={`nav-item ${
        active ? "active" : ""
      }`}
      onClick={onClick}
    >
      <span className="nav-icon">
        {icon}
      </span>

      <span className="nav-label">{label}</span>

      {badge !== undefined && (
        <span className="nav-badge">
          {badge}
        </span>
      )}
    </button>
  );
}

function ThemeSwitcher({ value, onChange }) {
  const options = [
    ["dark", Moon, "Dark"],
    ["light", Sun, "Light"],
    ["system", Monitor, "System"],
  ];
  return (
    <div className="theme-switcher" aria-label="Color theme">
      {options.map(([key, Icon, label]) => (
        <button key={key} type="button" className={value === key ? "active" : ""} onClick={() => onChange(key)} aria-label={`${label} theme`} aria-pressed={value === key}>
          <Icon size={14} />
        </button>
      ))}
    </div>
  );
}


/* ============================================================
   SEARCH PAGE
============================================================ */

function SearchPage({
  query,
  setQuery,
  results,
  loading,
  searched,
  error,
  memories,
  memoryCount,
  imageUrl,
  handleSearch,
  clearSearch,
  suggestedSearch,
  suggestions,
  predictiveSuggestions,
  recentSearches,
  sendFeedback,
  dashboardLoading,
}) {
  const searchInputRef = useRef(null);
  const [searchFocused, setSearchFocused] = useState(false);

  useEffect(() => {
    const focusSearch = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", focusSearch);
    return () => window.removeEventListener("keydown", focusSearch);
  }, []);

  return (
    <div className="page">

      <section className="hero">

        <div className="hero-glow" />

        <div className="eyebrow">
          <Sparkles size={14} />

          <span>
            YOUR VISUAL MEMORY
          </span>
        </div>

        <h1>
          Find anything
          <br />
          <span>
            you've ever seen.
          </span>
        </h1>

        <p>
          Search screenshots by
          meaning, context, and memory —
          not filenames.
        </p>

        <form
          className="search-box"
          onSubmit={handleSearch}
        >
          <Search
            size={21}
            className="search-box-icon"
          />

          <input
            ref={searchInputRef}
            value={query}
            onChange={(event) =>
              setQuery(
                event.target.value
              )
            }
            placeholder='Try "find my Python error"...'
            autoComplete="off"
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
          />

          {!searchFocused && !query && (
            <kbd className="search-shortcut" aria-label="Press Control K or Command K to focus search">Ctrl K</kbd>
          )}

          {query && (
            <button
              type="button"
              className="search-clear"
              onClick={clearSearch}
            >
              <X size={16} />
            </button>
          )}

          <button
            className="search-submit"
            type="submit"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2
                  size={15}
                  className="spin"
                />

                Searching
              </>
            ) : (
              <>
                Search

                <ArrowUpRight
                  size={16}
                />
              </>
            )}
          </button>
        </form>

        {(predictiveSuggestions.length > 0 || recentSearches.length > 0) && (
          <div className="predictive-search" aria-label="Search suggestions">
            {(predictiveSuggestions.length ? predictiveSuggestions : recentSearches).map((suggestion) => (
              <button type="button" key={suggestion} onClick={() => suggestedSearch(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <div className="quick-search">

          <span>
            Try searching
          </span>

          <button
            type="button"
            onClick={() =>
              suggestedSearch(
                "food"
              )
            }
          >
            🍔 Food
          </button>

          <button
            type="button"
            onClick={() =>
              suggestedSearch(
                "python"
              )
            }
          >
            🐍 Python
          </button>

          <button
            type="button"
            onClick={() =>
              suggestedSearch(
                "UPI payment"
              )
            }
          >
            💳 UPI
          </button>

          <button
            type="button"
            onClick={() =>
              suggestedSearch(
                "timetable"
              )
            }
          >
            📚 Timetable
          </button>

          <button
            type="button"
            onClick={() =>
              suggestedSearch(
                "everything"
              )
            }
          >
            ✨ Everything
          </button>
        </div>
      </section>

      {error && (
        <div className="error-card">
          <div>
            <strong>
              Something went wrong
            </strong>

            <span>
              {error}
            </span>
          </div>

          <button
            type="button"
            onClick={clearSearch}
          >
            <X size={16} />
          </button>
        </div>
      )}

      {searched ? (
        <section className="content-section">

          <SectionHeader
            kicker={
              query.toLowerCase() ===
              "everything"
                ? "ALL MEMORIES"
                : "SEMANTIC RESULTS"
            }
            title={
              loading
                ? "Searching your memory..."
                : `${results.length} memories found`
            }
            action={
              !loading ? (
                <button
                  type="button"
                  className="outline-button"
                  onClick={
                    clearSearch
                  }
                >
                  Back to memories
                </button>
              ) : null
            }
          />

          {loading ? (
            <LoadingGrid />
          ) : results.length ===
            0 ? (
            <EmptyState
              icon={
                <Search size={25} />
              }
              title="No matching memories"
              description="Try describing the screenshot differently."
            />
          ) : (
            <div className="memory-grid">
              {results.map(
                (memory, index) => (
                  <MemoryCard
                    key={
                      memory.memory_id ||
                      memory.id ||
                      index
                    }
                    memory={memory}
                    thumbnailUrl={imageUrl(
                      memory
                      , { thumbnail: true }
                    )}
                    originalImageUrl={imageUrl(memory)}
                    query={query}
                    sendFeedback={sendFeedback}
                    index={index}
                  />
                )
              )}
            </div>
          )}
        </section>
      ) : (
        <section className="content-section">

          <SectionHeader
            kicker="YOUR MEMORY"
            title="Recently indexed"
            action={
              <div className="memory-total">
                <Database size={15} />

                {memoryCount} memories
              </div>
            }
          />

          <div className="stats-grid">

            <Stat
              icon={
                <Database size={18} />
              }
              value={memoryCount}
              label="Memories indexed"
            />

            <Stat
              icon={
                <FileImage
                  size={18}
                />
              }
              value={memories.length}
              label="Screenshots"
            />

            <Stat
              icon={
                <FileText size={18} />
              }
              value="OCR"
              label="Text extraction"
            />

            <Stat
              icon={
                <Zap size={18} />
              }
              value="FAISS"
              label="Semantic retrieval"
            />
          </div>

          {dashboardLoading ? (
            <LoadingGrid />
          ) : memories.length >
            0 ? (
            <div className="memory-grid">
              {memories
                .slice(0, 12)
                .map(
                  (
                    memory,
                    index
                  ) => (
                    <MemoryCard
                      key={
                        memory.memory_id ||
                        memory.id ||
                        index
                      }
                    memory={memory}
                    thumbnailUrl={imageUrl(
                      memory
                      , { thumbnail: true }
                    )}
                    originalImageUrl={imageUrl(memory)}
                    index={index}
                    />
                  )
                )}
            </div>
          ) : (
            <EmptyState
              icon={
                <ImageIcon
                  size={25}
                />
              }
              title="Your visual memory is empty"
              description="Upload screenshots to start building your searchable memory."
            />
          )}

          {memories.length >
            0 && (
            <SuggestionBar
            suggestions={
              suggestions
            }
              onSearch={
                suggestedSearch
              }
            />
          )}
        </section>
      )}
    </div>
  );
}


/* ============================================================
   MEMORIES PAGE
============================================================ */

function MemoriesPage({
  memories,
  memoryCount,
  imageUrl,
  loading,
  hasMoreMemories,
  loadMoreMemories,
  categories,
  activeCategory,
  activeSort,
  updateLibraryFilter,
}) {
  return (
    <div className="page">

      <section className="page-title">

        <div>
          <div className="eyebrow small">
            <Layers3 size={14} />

            MEMORY LIBRARY
          </div>

          <h1>
            Your memories
          </h1>

          <p>
            Every indexed screenshot in
            one place.
          </p>
        </div>

        <div className="big-count">
          <strong>
            {memoryCount}
          </strong>

          <span>
            memories
          </span>
        </div>
      </section>

      <div className="library-controls">
        <select value={activeCategory} onChange={(event) => updateLibraryFilter(event.target.value)} aria-label="Filter memories by category">
          <option value="">Everything</option>
          {categories.map((category) => <option key={category} value={category}>{category}</option>)}
        </select>
        <select value={activeSort} onChange={(event) => updateLibraryFilter(activeCategory, event.target.value)} aria-label="Sort memories">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
      </div>

      {loading ? (
        <LoadingGrid />
      ) : memories.length ? (
        <>
          <div className="memory-grid large">

          {memories.map(
            (memory, index) => (
              <MemoryCard
                key={
                  memory.memory_id ||
                  memory.id ||
                  index
                }
                memory={memory}
                thumbnailUrl={imageUrl(
                  memory
                  , { thumbnail: true }
                )}
                originalImageUrl={imageUrl(memory)}
                index={index}
              />
            )
          )}
          </div>
          {hasMoreMemories && (
            <div className="load-more-wrap">
              <button type="button" className="load-more" onClick={loadMoreMemories}>
                Load more memories
              </button>
            </div>
          )}
        </>
      ) : (
        <EmptyState
          icon={
            <Layers3 size={25} />
          }
          title="No memories yet"
          description="Upload screenshots and they'll appear here."
        />
      )}
    </div>
  );
}


/* ============================================================
   ANALYTICS PAGE
============================================================ */

function AnalyticsPage({
  memoryCount,
  imageCount,
  memories,
}) {
  const categories = {};

  memories.forEach(
    (memory) => {
      const category =
        memory?.category ||
        memory?.type ||
        "Other";

      categories[category] =
        (categories[category] ||
          0) + 1;
    }
  );

  const categoryList =
    Object.entries(categories)
      .sort(
        (a, b) => b[1] - a[1]
      )
      .slice(0, 6);

  return (
    <div className="page">

      <section className="page-title">

        <div>
          <div className="eyebrow small">
            <BarChart3 size={14} />

            MEMORY ANALYTICS
          </div>

          <h1>
            Memory intelligence
          </h1>

          <p>
            Understand how your visual
            memory is growing.
          </p>
        </div>
      </section>

      <div className="analytics-grid">

        <AnalyticsCard
          label="Total memories"
          value={memoryCount}
          icon={
            <Database size={20} />
          }
        />

        <AnalyticsCard
          label="Indexed images"
          value={imageCount}
          icon={
            <ImageIcon size={20} />
          }
        />

        <AnalyticsCard
          label="Search engine"
          value="FAISS"
          icon={
            <Zap size={20} />
          }
        />

        <AnalyticsCard
          label="AI pipeline"
          value="ONLINE"
          icon={
            <Activity size={20} />
          }
        />
      </div>

      <div className="analytics-panel">

        <div className="panel-header">

          <div>
            <span>
              MEMORY DISTRIBUTION
            </span>

            <h2>
              What you've been saving
            </h2>
          </div>

          <div className="live-label">
            <span />
            Live
          </div>
        </div>

        {categoryList.length ? (
          <div className="category-list">

            {categoryList.map(
              ([category, count]) => {

                const percentage =
                  memoryCount
                    ? Math.round(
                        (count /
                          memoryCount) *
                          100
                      )
                    : 0;

                return (
                  <div
                    className="category-row"
                    key={category}
                  >
                    <div className="category-name">
                      <span />
                      {category}
                    </div>

                    <div className="category-bar">
                      <div
                        style={{
                          width: `${percentage}%`,
                        }}
                      />
                    </div>

                    <strong>
                      {count}
                    </strong>
                  </div>
                );
              }
            )}
          </div>
        ) : (
          <div className="analytics-empty">
            Upload screenshots to see
            memory analytics.
          </div>
        )}
      </div>

      <div className="pipeline">

        <PipelineStep
          number="01"
          title="Capture"
          description="Screenshots enter MemoryOS."
          icon={
            <ImageIcon size={18} />
          }
        />

        <PipelineStep
          number="02"
          title="OCR"
          description="Text is extracted from images."
          icon={
            <FileText size={18} />
          }
        />

        <PipelineStep
          number="03"
          title="AI"
          description="Content is understood semantically."
          icon={
            <Sparkles size={18} />
          }
        />

        <PipelineStep
          number="04"
          title="Retrieve"
          description="FAISS finds the right memory."
          icon={
            <Search size={18} />
          }
        />
      </div>
    </div>
  );
}


/* ============================================================
   UPLOAD MODAL
============================================================ */

function UploadModal({
  files,
  uploading,
  status,
  fileInputRef,
  addFiles,
  removeFile,
  upload,
  close,
}) {
  const [dragging, setDragging] =
    useState(false);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (
          event.target ===
          event.currentTarget
        ) {
          close();
        }
      }}
    >
      <div className="upload-modal">

        <div className="modal-header">

          <div>
            <div className="modal-icon">
              <Upload size={19} />
            </div>

            <div>
              <span>
                MEMORY INGESTION
              </span>

              <h2>
                Upload screenshots
              </h2>
            </div>
          </div>

          <button
            type="button"
            className="icon-button"
            onClick={close}
            disabled={uploading}
          >
            <X size={19} />
          </button>
        </div>

        <div
          className={`drop-zone ${
            dragging
              ? "dragging"
              : ""
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() =>
            setDragging(false)
          }
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);

            addFiles(
              event.dataTransfer.files
            );
          }}
          onClick={() =>
            fileInputRef.current?.click()
          }
        >

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            hidden
            onChange={(event) => {
              addFiles(
                event.target.files
              );

              event.target.value =
                "";
            }}
          />

          <div className="drop-icon">
            <Upload size={23} />
          </div>

          <h3>
            Drop screenshots here
          </h3>

          <p>
            or click to browse your
            computer
          </p>

          <div className="drop-meta">
            PNG · JPG · WEBP · up to
            10 MB each
          </div>
        </div>

        {files.length > 0 && (
          <div className="upload-files">

            <div className="upload-files-header">

              <strong>
                {files.length} selected
              </strong>

              <span>
                Ready for indexing
              </span>
            </div>

            <div className="file-list">

              {files.map(
                (file, index) => (
                  <div
                    className="file-row"
                    key={`${file.name}-${index}`}
                  >
                    <div className="file-preview">
                      <FileImage
                        size={17}
                      />
                    </div>

                    <div className="file-info">
                      <strong>
                        {file.name}
                      </strong>

                      <span>
                        {formatBytes(
                          file.size
                        )}
                      </span>
                    </div>

                    {!uploading && (
                      <button
                        type="button"
                        onClick={() =>
                          removeFile(
                            index
                          )
                        }
                      >
                        <Trash2
                          size={15}
                        />
                      </button>
                    )}
                  </div>
                )
              )}
            </div>
          </div>
        )}

        {status && (
          <div className="upload-status">

            {uploading ? (
              <Loader2
                size={17}
                className="spin"
              />
            ) : (
              <Check size={17} />
            )}

            <span>
              {status}
            </span>
          </div>
        )}

        <div className="modal-footer">

          <span>
            <Zap size={14} />

            AI indexing starts
            automatically
          </span>

          <button
            type="button"
            className="primary-button"
            disabled={
              !files.length ||
              uploading
            }
            onClick={upload}
          >
            {uploading ? (
              <>
                <Loader2
                  size={16}
                  className="spin"
                />

                Processing...
              </>
            ) : (
              <>
                <Upload size={16} />

                Index screenshots
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}


/* ============================================================
   MEMORY CARD
============================================================ */

function MemoryCard({
  memory,
  thumbnailUrl,
  originalImageUrl,
  query = "",
  sendFeedback,
  index = 0,
}) {
  const [failed, setFailed] =
    useState(false);

  const [loaded, setLoaded] =
    useState(false);

  const [viewerOpen, setViewerOpen] =
    useState(false);

  const filename =
    memory?.filename ||
    memory?.file_name ||
    memory?.original_filename ||
    memory?.name ||
    "Untitled screenshot";

  const category =
    memory?.category ||
    memory?.type ||
    "Memory";

  const vision = memory?.vision_analysis && typeof memory.vision_analysis === "object" ? memory.vision_analysis : {};
  const visualDescription = memory?.visual_description || vision.visual_description || memory?.summary || memory?.description || "";
  const visualConcepts = [
    ...(memory?.visual_concepts || vision.visual_concepts || []),
    ...(memory?.food_concepts || vision.food_concepts || []),
    ...(memory?.objects || vision.objects || []),
    ...(memory?.general_concepts || vision.general_concepts || []),
  ].filter(Boolean).map(String).filter((value, index, values) => values.indexOf(value) === index).slice(0, 5);

  // OCR is deliberately secondary. A card's primary copy only uses the
  // stored visual description or summary, never manufactured text.
  const description = visualDescription || memory?.summary || "Image indexed by MemoryOS.";

  const matchKinds = Array.isArray(memory?.why_matched)
    ? Array.from(new Set(memory.why_matched.map((reason) => {
      const value = String(reason).toLowerCase();
      if (/visual concept|visual meaning/.test(value)) return "Visual";
      if (/semantic|meaning/.test(value)) return "Semantic";
      if (/text|ocr|phrase/.test(value)) return "OCR";
      if (/category|keyword|entity/.test(value)) return "Concept";
      if (/filename/.test(value)) return "Filename";
      return null;
    }).filter(Boolean)))
    : [];

  const score =
    typeof memory?.score ===
    "number"
      ? memory.score > 1
        ? memory.score
        : memory.score * 100
      : null;

  return (
    <article className="memory-card" style={{ "--result-index": Math.min(index, 8) }}>

      <button
        type="button"
        className="card-image"
        onClick={() => {
          if (originalImageUrl && !failed) {
            setViewerOpen(true);
          }
        }}
        aria-label={`Open original screenshot: ${filename}`}
        disabled={!originalImageUrl || failed}
      >

        {thumbnailUrl && !failed ? (
          <>
            {!loaded && (
              <div className="image-loading">
                <Loader2
                  size={25}
                  className="spin"
                />

                <span>
                  Loading preview...
                </span>
              </div>
            )}

            <img
              src={thumbnailUrl}
              alt={filename}
              loading="lazy"
              decoding="async"
              onLoad={() =>
                setLoaded(true)
              }
              onError={(event) => {
                console.error(
                  "Image preview failed:",
                  thumbnailUrl,
                  event
                );

                setFailed(true);
                setLoaded(false);
              }}
            />
          </>
        ) : (
          <div className="image-fallback">
            <ImageIcon size={28} />

            <span>
              Preview unavailable
            </span>

            {memory?.memory_id && (
              <small>
                {String(
                  memory.memory_id
                ).slice(0, 12)}
              </small>
            )}
          </div>
        )}

        <div className="image-overlay" />

        {score !== null && (
          <div className="match">
            {Math.round(score)}%
            match
          </div>
        )}

        <div className="image-category">
          {category}
        </div>

        {matchKinds.includes("Visual") && (
          <div className="visual-match-badge">Visual match</div>
        )}
      </button>

      <div className="card-body">

        <div className="card-title-row">

          <h3 title={filename}>
            {filename}
          </h3>

          <ArrowUpRight
            size={15}
          />
        </div>

        <p className="card-description">
          {String(
            description
          ).slice(0, 150)}

          {String(description)
            .length > 150
            ? "..."
            : ""}
        </p>

        {visualConcepts.length > 0 && (
          <div className="concept-chips" aria-label="Visual concepts">
            {visualConcepts.map((concept) => <span key={concept}><Sparkles size={11} />{concept}</span>)}
          </div>
        )}

        {matchKinds.length > 0 && (
          <div className="match-kind-chips" aria-label="Match evidence">
            {matchKinds.map((kind) => <span key={kind}>{kind} match</span>)}
          </div>
        )}

        {Array.isArray(memory?.why_matched) && memory.why_matched.length > 0 && (
          <details className="memory-details match-details">
            <summary>Why this matched</summary>
            <ul>{memory.why_matched.slice(0, 4).map((reason) => <li key={reason}>{reason}</li>)}</ul>
          </details>
        )}

        {(visualDescription || visualConcepts.length > 0) && (
          <details className="memory-details">
            <summary>AI Visual Understanding</summary>
            {visualDescription && <p>{visualDescription}</p>}
            {visualConcepts.length > 0 && <div className="concept-chips">{visualConcepts.map((concept) => <span key={`detail-${concept}`}>{concept}</span>)}</div>}
          </details>
        )}

        <PhotoOcr text={memory?.ocr_text} confidence={memory?.ocr_confidence ?? memory?.confidence} />

        {Array.isArray(memory?.why_matched) && memory.why_matched.length > 0 && (
          <div className="match-reason">{memory.why_matched[0]}</div>
        )}

        {sendFeedback && query && memory?.memory_id && (
          <div className="result-feedback">
            <button type="button" onClick={() => sendFeedback(memory.memory_id, true)}>Relevant</button>
            <button type="button" onClick={() => sendFeedback(memory.memory_id, false)}>Not relevant</button>
          </div>
        )}

        <div className="card-footer">

          <span>
            <Clock3 size={13} />

            Indexed memory
          </span>

          <span className="indexed-dot">
            <span />

            Indexed
          </span>
        </div>
      </div>

      {viewerOpen && (
        <ImageViewer
          imageUrl={originalImageUrl}
          filename={filename}
          memory={memory}
          close={() => setViewerOpen(false)}
        />
      )}
    </article>
  );
}

function PhotoOcr({ text, confidence }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (!text || !navigator.clipboard) return;
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };
  return (
    <details className="memory-details ocr-details">
      <summary>Photo OCR{text ? "" : " · no readable text"}</summary>
      {text ? <><pre>{text}</pre><div className="ocr-actions"><span>{typeof confidence === "number" ? `${Math.round(confidence * (confidence <= 1 ? 100 : 1))}% confidence` : "Extracted text"}</span><button type="button" onClick={copy}><Copy size={12} />{copied ? "Copied" : "Copy"}</button></div></> : <p>No readable text detected.</p>}
    </details>
  );
}


function ImageViewer({
  imageUrl,
  filename,
  memory,
  close,
}) {
  const [failed, setFailed] = useState(false);
  const [related, setRelated] = useState([]);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape") close();
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [close]);

  useEffect(() => {
    if (!memory?.memory_id) return;
    fetch(`${API}/search/related/${encodeURIComponent(memory.memory_id)}?limit=4`)
      .then((response) => response.ok ? response.json() : { results: [] })
      .then((data) => setRelated(Array.isArray(data?.results) ? data.results : []))
      .catch(() => setRelated([]));
  }, [memory?.memory_id]);

  return (
    <div
      className="image-viewer-backdrop"
      role="presentation"
      onMouseDown={close}
    >
      <section
        className="image-viewer"
        role="dialog"
        aria-modal="true"
        aria-label={`Original screenshot: ${filename}`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="image-viewer-close"
          onClick={close}
          aria-label="Close screenshot viewer"
        >
          <X size={20} />
        </button>

        <div className="viewer-zoom-controls" aria-label="Image zoom controls">
          <button type="button" onClick={() => setZoom((value) => Math.max(0.5, value - 0.25))} aria-label="Zoom out"><ZoomOut size={17} /></button>
          <button type="button" onClick={() => setZoom(1)} aria-label="Reset zoom"><RotateCcw size={16} /></button>
          <button type="button" onClick={() => setZoom((value) => Math.min(3, value + 0.25))} aria-label="Zoom in"><ZoomIn size={17} /></button>
        </div>

        {failed ? (
          <div className="image-viewer-error">
            The original screenshot is unavailable.
          </div>
        ) : (
          <img
            src={imageUrl}
            alt={filename}
            onError={() => setFailed(true)}
            style={{ transform: `scale(${zoom})` }}
          />
        )}
        <aside className="viewer-details">
          <strong>{filename}</strong>
          {memory?.visual_description && <p>{memory.visual_description}</p>}
          {(memory?.keywords || []).slice(0, 8).join(" · ")}
          {memory?.summary && <p>{memory.summary}</p>}
          {memory?.ocr_text && <p className="viewer-ocr">{memory.ocr_text.slice(0, 180)}</p>}
          {related.length > 0 && <p>Related memories: {related.map((item) => item.filename || item.memory_id).join(", ")}</p>}
        </aside>
      </section>
    </div>
  );
}


/* ============================================================
   SECTION HEADER
============================================================ */

function SectionHeader({
  kicker,
  title,
  action,
}) {
  return (
    <div className="section-header">

      <div>
        <span className="section-kicker">
          {kicker}
        </span>

        <h2>
          {title}
        </h2>
      </div>

      {action}
    </div>
  );
}


/* ============================================================
   STAT
============================================================ */

function Stat({
  icon,
  value,
  label,
}) {
  return (
    <div className="stat">

      <div className="stat-icon">
        {icon}
      </div>

      <div>
        <strong>
          {value}
        </strong>

        <span>
          {label}
        </span>
      </div>
    </div>
  );
}


/* ============================================================
   ANALYTICS CARD
============================================================ */

function AnalyticsCard({
  icon,
  value,
  label,
}) {
  return (
    <div className="analytics-card">

      <div className="analytics-card-icon">
        {icon}
      </div>

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

      <div className="analytics-card-line" />
    </div>
  );
}


/* ============================================================
   PIPELINE
============================================================ */

function PipelineStep({
  number,
  title,
  description,
  icon,
}) {
  return (
    <div className="pipeline-step">

      <span className="pipeline-number">
        {number}
      </span>

      <div className="pipeline-icon">
        {icon}
      </div>

      <div>
        <strong>
          {title}
        </strong>

        <span>
          {description}
        </span>
      </div>
    </div>
  );
}


/* ============================================================
   SUGGESTION BAR
============================================================ */

function SuggestionBar({
  suggestions,
  onSearch,
}) {
  return (
    <div className="suggestion-bar">

      <div className="suggestion-heading">

        <Sparkles size={16} />

        <div>
          <strong>
            Explore your memory
          </strong>

          <span>
            Suggested searches
          </span>
        </div>
      </div>

      <div className="suggestion-items">

        {suggestions.map(
          (suggestion, index) => (
            <button
              type="button"
              key={`${suggestion}-${index}`}
              onClick={() =>
                onSearch(
                  suggestion
                )
              }
            >
              <Search size={13} />

              {suggestion}
            </button>
          )
        )}
      </div>
    </div>
  );
}


/* ============================================================
   EMPTY STATE
============================================================ */

function EmptyState({
  icon,
  title,
  description,
}) {
  return (
    <div className="empty">

      <div className="empty-icon">
        {icon}
      </div>

      <h3>
        {title}
      </h3>

      <p>
        {description}
      </p>
    </div>
  );
}


/* ============================================================
   LOADING GRID
============================================================ */

function LoadingGrid() {
  return (
    <div className="memory-grid">

      {Array.from({
        length: 6,
      }).map((_, index) => (
        <div
          className="skeleton-card"
          key={index}
        >
          <div className="skeleton-image" />

          <div className="skeleton-body">
            <div />
            <div />
            <div />
          </div>
        </div>
      ))}
    </div>
  );
}


/* ============================================================
   FORMAT BYTES
============================================================ */

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }

  const units = [
    "B",
    "KB",
    "MB",
    "GB",
  ];

  const index = Math.min(
    Math.floor(
      Math.log(bytes) /
        Math.log(1024)
    ),
    units.length - 1
  );

  return `${(
    bytes /
    Math.pow(1024, index)
  ).toFixed(
    index ? 1 : 0
  )} ${units[index]}`;
}


export default App;
