# Collapsible navigation sidebar template
<aside class="sidebar">
    <div class="sidebar-top">
        @panel(heading="Navigation", elevation=1, collapsible=true)
    </div>
    <ul class="sidebar-menu">
        <li class="menu-entry">Cluster Nodes</li>
        <li class="menu-entry">Storage Volumes</li>
        <li class="menu-entry">Network Policies</li>
        <li class="menu-entry">Audit Trail</li>
    </ul>
    <div class="sidebar-bottom">
        @badge(label="healthy", level=1)
        @grid(cols=2, gap=6, title="SidebarMiniGrid")
    </div>
</aside>
