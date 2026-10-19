# Global navigation header template
<header class="site-header">
    <div class="brand-bar">
        <h1 class="site-title">Telemetry Console</h1>
        @badge(label="production", level=3)
    </div>
    <nav class="primary-nav">
        <a class="nav-item" href="/overview">Overview</a>
        <a class="nav-item" href="/pipelines">Pipelines</a>
        <a class="nav-item" href="/alerts">Alerts</a>
        @panel(heading="Quick Status", elevation=1, collapsible=false)
    </nav>
</header>
