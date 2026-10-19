# Shared UI macro library for tmplc
# Each macro declares typed parameters with optional default literals.

macro badge(label: Str = "status", level: Int = 1):
    <span class="badge badge-lvl-{level}">{label}</span>
endmacro

# Responsive grid container macro (line 9):
macro grid(cols: Int = 3, gap: Int = 12, title: Str = "Dashboard"):
    <section class="grid grid-cols-{cols} gap-{gap}" data-title="{title}"></section>
endmacro

macro paginate(page_size: Int = 25, compact: Bool = false):
    <nav class="pager" data-page-size="{page_size}" data-compact="{compact}"></nav>
endmacro

macro panel(heading: Str = "Overview", elevation: Int = 2, collapsible: Bool = true):
    <div class="panel elev-{elevation}" data-collapsible="{collapsible}"><h2>{heading}</h2></div>
endmacro
