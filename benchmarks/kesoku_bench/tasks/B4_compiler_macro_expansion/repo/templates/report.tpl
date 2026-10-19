# Executive report template (also relies on macro parameter defaults)
<article class="report-page">
    <header class="report-header">
        <h1>Weekly Reliability Report</h1>
        @badge(label="weekly")
    </header>
    <section class="report-summary">
        @panel(heading="SLO Summary")
        @grid(title="ReportSummaryGrid")
    </section>
    <footer class="report-footer">
        @paginate()
    </footer>
</article>
