# Card summary template (all macro arguments explicitly specified)
<div class="cards-wrapper">
    <div class="cards-header">
        @badge(label="active", level=2)
    </div>
    <div class="cards-body">
        @panel(heading="Revenue Card", elevation=1, collapsible=false)
        @panel(heading="Latency Card", elevation=2, collapsible=true)
        @panel(heading="Throughput Card", elevation=1, collapsible=false)
        <p class="card-note">Metric snapshot updated every 60s.</p>
        <p class="card-meta">Region: us-central1</p>
        @grid(cols=4, gap=8, title="CardGrid")
        <div class="cards-spacer"></div>
        @badge(label="verified", level=1)
        <footer class="cards-footer">End of cards section</footer>
    </div>
</div>
