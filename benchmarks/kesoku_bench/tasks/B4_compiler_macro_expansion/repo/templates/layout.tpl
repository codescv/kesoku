# Primary application layout template
# Combines header, sidebar, main content grid, and footer regions.
#
# Structure overview:
#   - Top banner & breadcrumb bar
#   - Workspace shell container
#   - Main content grid (uses default cols from _macros.tpl)
#   - Status bar & pagination footer
#
# Notes:
#   - Keep indentation aligned for static DOM inspection.
#   - Macro defaults are resolved during tmplc expansion.
<div class="app-layout">
    <div class="layout-top-bar">
        <span class="crumb">Home</span>
        <span class="crumb-sep">/</span>
        <span class="crumb-active">Dashboard</span>
        @badge(label="live", level=2)
    </div>
    <div class="layout-shell">
        <div class="layout-left-rail">
            @panel(heading="Filters", elevation=2, collapsible=true)
            <ul class="filter-list">
                <li class="filter-item">Environment: prod</li>
                <li class="filter-item">Service: ingest-gw</li>
                <li class="filter-item">Window: 24h</li>
                <li class="filter-item">Percentile: p99</li>
            </ul>
        </div>
        <main class="layout-main">
            <div class="layout-hero">
                <h2 class="hero-heading">Cluster Health Matrix</h2>
                <p class="hero-sub">Real-time shard allocation and queue depth overview.</p>
            </div>
            <div class="layout-kpi-strip">
                <div class="kpi-box">Nodes Online: 64</div>
                <div class="kpi-box">Active Streams: 512</div>
                <div class="kpi-box">Error Budget: 99.94%</div>
                <div class="kpi-box">Replication Lag: 14ms</div>
            </div>
            <div class="layout-content-region">
                <div class="region-toolbar">
                    <span class="toolbar-title">Shard Distribution</span>
                    @badge(label="auto-refresh", level=1)
                </div>
                # Main dashboard grid relies on default cols=3 from _macros.tpl:9
                <div class="region-grid-host">
                @grid(gap=16, title="Main")
                </div>
                <div class="region-pager-host">
                    @paginate(page_size=50, compact=false)
                </div>
            </div>
        </main>
    </div>
</div>
