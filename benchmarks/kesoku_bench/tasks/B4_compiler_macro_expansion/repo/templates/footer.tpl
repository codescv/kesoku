# Global page footer template
<footer class="site-footer">
    <div class="footer-inner">
        <p class="copyright">Copyright 2026 Platform Engineering</p>
        @badge(label="v2.4.0", level=1)
        <ul class="footer-links">
            <li><a href="/status">System Status</a></li>
            <li><a href="/docs">Documentation</a></li>
            <li><a href="/support">Support</a></li>
        </ul>
        @paginate(page_size=10, compact=true)
    </div>
</footer>
