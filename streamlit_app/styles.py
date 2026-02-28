"""
EnsAgent Premium UI Styles
Modern SaaS analytics dashboard aesthetic — clean, airy, polished.
"""
import html

COLORS = {
    "background": "#F5F5F7",
    "background_secondary": "#EDEDF0",
    "surface": "#FEFEFE",
    "border": "#E9E9EB",
    "border_light": "#E9E9EB",
    "text_primary": "#181A1F",
    "text_secondary": "#6E6F78",
    "text_tertiary": "#8A8D98",
    "accent": "#5A3683",
    "accent_hover": "#6E44A0",
    "accent_light": "#EDE8F3",
    "success": "#34C759",
    "warning": "#F5A623",
    "error": "#E74C3C",
    "shadow": "rgba(0, 0, 0, 0.04)",
    "shadow_hover": "rgba(0, 0, 0, 0.07)",
    "chart_blue": "#6B8FD4",
    "chart_blue_light": "#A3BFE8",
    "chart_blue_pale": "#D4E3F8",
}

FONTS = {
    "primary": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "display": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "mono": "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
}


def get_premium_css() -> str:
    """Return the complete premium CSS stylesheet."""
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --bg-primary: {COLORS["background"]};
            --bg-secondary: {COLORS["background_secondary"]};
            --surface: {COLORS["surface"]};
            --border: {COLORS["border"]};
            --border-light: {COLORS["border_light"]};
            --text-primary: {COLORS["text_primary"]};
            --text-secondary: {COLORS["text_secondary"]};
            --text-tertiary: {COLORS["text_tertiary"]};
            --accent: {COLORS["accent"]};
            --accent-hover: {COLORS["accent_hover"]};
            --accent-light: {COLORS["accent_light"]};
            --success: {COLORS["success"]};
            --warning: {COLORS["warning"]};
            --error: {COLORS["error"]};
            --shadow: {COLORS["shadow"]};
            --shadow-hover: {COLORS["shadow_hover"]};
            --radius-sm: 8px;
            --radius-md: 10px;
            --radius-lg: 12px;
            --radius-xl: 16px;
            --font-primary: {FONTS["primary"]};
            --font-display: {FONTS["display"]};
            --font-mono: {FONTS["mono"]};
        }}

        /* ===== Global Reset & Base ===== */
        .stApp,
        [data-testid="stAppViewContainer"] {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-primary);
        }}

        #MainMenu, footer {{
            visibility: hidden;
        }}

        button[data-testid="baseButton-header"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        button[aria-label="Collapse sidebar"],
        button[aria-label="Expand sidebar"] {{
            display: inline-flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }}

        header [data-testid="collapsedControl"],
        div[data-testid="stToolbar"] [data-testid="collapsedControl"],
        header button[data-testid="baseButton-header"],
        header [data-testid="stSidebarCollapseButton"],
        div[data-testid="stToolbar"] [data-testid="stSidebarCollapseButton"],
        header [data-testid="stSidebarCollapsedControl"],
        header button[aria-label="Collapse sidebar"],
        header button[aria-label="Expand sidebar"] {{
            position: relative !important;
            z-index: 9999 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}

        /* Keep Streamlit Material Symbols rendering as icons, not plain text. */
        [data-testid="collapsedControl"] span,
        button[data-testid="baseButton-header"] span {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            font-size: 0 !important;
            line-height: 1 !important;
        }}

        [data-testid="collapsedControl"] span::before,
        button[data-testid="baseButton-header"] span::before {{
            content: "keyboard_double_arrow_right";
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
            font-size: 1.2rem !important;
            line-height: 1 !important;
            display: inline-block;
            color: var(--text-primary);
        }}

        /* ===== Typography ===== */
        html, body, div, p, a, button, input, textarea, select, label {{
            font-family: var(--font-primary) !important;
        }}

        .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stSlider {{
            font-family: var(--font-primary) !important;
        }}

        h1 {{
            font-size: 1.75rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em !important;
            line-height: 1.2 !important;
            font-family: var(--font-primary) !important;
        }}

        h2 {{
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em !important;
            font-family: var(--font-primary) !important;
        }}

        h3 {{
            font-size: 0.8125rem !important;
            font-weight: 600 !important;
            color: var(--text-tertiary) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }}

        p, li {{
            color: var(--text-secondary);
            line-height: 1.55;
            letter-spacing: -0.006em;
        }}

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.25rem;
            padding-bottom: 1rem;
        }}

        @media (min-width: 992px) {{
            section[data-testid="stSidebar"] > div {{
                padding-top: 1rem;
                padding-bottom: 1rem;
            }}
        }}

        section[data-testid="stSidebar"] .stMarkdown h1 {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}

        section[data-testid="stSidebar"] .stMarkdown h3 {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 1.25rem;
            margin-bottom: 0.4rem;
        }}

        /* Sidebar nav buttons */
        section[data-testid="stSidebar"] .stButton > button {{
            background: transparent !important;
            color: var(--text-secondary) !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            padding: 0.52rem 0.74rem !important;
            font-weight: 500 !important;
            font-size: 0.78rem !important;
            text-align: left !important;
            box-shadow: none !important;
            transition: background 0.15s ease, color 0.15s ease !important;
            justify-content: flex-start !important;
        }}

        section[data-testid="stSidebar"] .stButton > button:hover {{
            background: #F0F0F2 !important;
            color: var(--text-primary) !important;
            transform: none !important;
            box-shadow: none !important;
        }}

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: var(--accent-light) !important;
            color: var(--accent) !important;
            border: 1px solid transparent !important;
            font-weight: 600 !important;
        }}

        [style*="col-resize"] {{
            cursor: default !important;
        }}

        div[style*="cursor: col-resize"][style*="position: absolute"] {{
            display: none !important;
            pointer-events: none !important;
        }}

        /* ===== Page Header ===== */
        .ens-header {{
            padding-bottom: 0.75rem;
            margin-bottom: 1.25rem;
        }}

        .ens-page-title {{
            font-family: var(--font-primary);
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.025em;
            margin: 0 0 0.15rem 0;
        }}

        .ens-page-subtitle {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: 0;
        }}

        .ens-section {{
            margin: 0 0 1rem 0;
        }}

        .ens-section-title {{
            font-family: var(--font-primary);
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.025em;
            margin-bottom: 0.15rem;
        }}

        .ens-section-subtitle {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: 0;
        }}

        /* ===== Sidebar Brand ===== */
        .ens-sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.25rem 0.5rem 1rem 0.5rem;
        }}

        .ens-sidebar-icon {{
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.1rem;
            color: var(--text-primary);
            flex-shrink: 0;
        }}

        .ens-sidebar-title {{
            font-size: 1.14rem;
            font-weight: 600;
            margin: 0;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }}

        .ens-sidebar-subtitle {{
            font-size: 0.68rem;
            color: var(--text-secondary);
            margin: 0;
            letter-spacing: 0.02em;
        }}

        /* Sidebar nav item with icon */
        .ens-nav-item {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
            transition: background 0.15s ease, color 0.15s ease;
            margin-bottom: 0.125rem;
        }}

        .ens-nav-item:hover {{
            background: #F0F0F2;
            color: var(--text-primary);
        }}

        .ens-nav-item.active {{
            background: var(--accent-light);
            color: var(--accent);
            font-weight: 600;
        }}

        .ens-nav-icon {{
            width: 18px;
            text-align: center;
            font-size: 0.875rem;
            opacity: 0.7;
        }}

        .ens-nav-item.active .ens-nav-icon {{
            opacity: 1;
        }}

        /* ===== Filter Chips ===== */
        .ens-filters {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            flex-wrap: wrap;
        }}

        .ens-filter-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.375rem 0.875rem;
            font-size: 0.8125rem;
            font-weight: 500;
            color: var(--text-primary);
            cursor: pointer;
            transition: border-color 0.15s ease;
        }}

        .ens-filter-chip:hover {{
            border-color: var(--text-tertiary);
        }}

        .ens-filter-chip svg, .ens-filter-chip .chip-icon {{
            width: 14px;
            height: 14px;
            opacity: 0.5;
        }}

        .ens-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
        }}

        .ens-chip {{
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        .ens-chip-row-spaced {{
            margin-top: 0.5rem;
        }}

        .ens-chip-actions-anchor {{
            display: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.ens-chip-actions-anchor) {{
            max-width: 1080px;
            margin: 0.35rem auto 0.85rem auto;
            padding: 0.35rem 0.45rem 0.65rem;
            background: transparent;
            border: none;
            border-radius: 0;
            box-shadow: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.ens-chip-actions-anchor) .stButton > button {{
            background: var(--bg-primary) !important;
            color: var(--text-secondary) !important;
            border: 2px solid var(--border) !important;
            border-radius: 999px !important;
            padding: 0.375rem 0.875rem !important;
            font-size: 0.8125rem !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }}

        div[data-testid="stVerticalBlock"]:has(.ens-chip-actions-anchor) .stButton > button:hover {{
            border-color: var(--text-tertiary) !important;
            background: var(--bg-secondary) !important;
            transform: none !important;
        }}

        /* ===== Cards ===== */
        .ens-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px var(--shadow);
            transition: box-shadow 0.15s ease;
        }}

        .ens-card:hover {{
            box-shadow: 0 2px 8px var(--shadow-hover);
        }}

        /* ===== KPI / Metrics ===== */
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem 1.125rem;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        div[data-testid="stMetric"] label {{
            color: var(--text-tertiary) !important;
            font-size: 0.6875rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }}

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }}

        /* ===== Buttons ===== */
        .stButton > button {{
            background: var(--accent) !important;
            color: white !important;
            border: 1px solid var(--accent) !important;
            border-radius: 999px !important;
            padding: 0.5rem 1.25rem !important;
            font-weight: 500 !important;
            font-size: 0.8125rem !important;
            letter-spacing: -0.006em !important;
            transition: background 0.15s ease, box-shadow 0.15s ease !important;
            box-shadow: 0 1px 3px var(--shadow) !important;
        }}

        .stButton > button:hover {{
            background: var(--accent-hover) !important;
            box-shadow: 0 2px 8px var(--shadow-hover) !important;
            transform: none !important;
        }}

        .stButton > button:active {{
            transform: none !important;
        }}

        .stButton > button[kind="secondary"] {{
            background: var(--surface) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
        }}

        .stButton > button[kind="secondary"]:hover {{
            background: var(--bg-primary) !important;
        }}

        /* ===== Input Fields ===== */
        .stTextInput input,
        .stTextArea textarea {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            font-size: 0.875rem !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            outline: none !important;
        }}

        .stTextInput > div,
        .stTextInput > div > div,
        .stTextArea > div,
        .stTextArea > div > div {{
            border: none !important;
            box-shadow: none !important;
        }}

        div[data-testid="stTextInput"] {{
            margin-bottom: 0.9rem !important;
        }}

        div[data-testid="InputInstructions"] {{
            position: static !important;
            margin-top: 0.25rem !important;
            color: var(--text-tertiary) !important;
            line-height: 1.2 !important;
        }}

        div[data-testid="InputInstructions"] p {{
            margin: 0 !important;
            font-size: 0.7rem !important;
        }}

        .stTextInput input:focus,
        .stTextArea textarea:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(90, 54, 131, 0.12) !important;
        }}

        .stSelectbox > div > div {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-md) !important;
        }}

        /* Chat input */
        div[data-testid="stChatInput"] {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 0.35rem 0.625rem !important;
            box-shadow: 0 1px 3px var(--shadow) !important;
        }}

        div[data-testid="stChatInput"] textarea {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }}

        /* ===== Pipeline Stages ===== */
        .ens-stage {{
            display: flex;
            align-items: center;
            gap: 0.875rem;
            padding: 0.875rem 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        .ens-stage[data-active="true"] {{
            border-left: 3px solid var(--accent);
        }}

        .ens-stage-icon {{
            width: 36px;
            height: 36px;
            background: var(--bg-primary);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8125rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        .ens-stage-body {{
            flex: 1;
        }}

        .ens-stage-title {{
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.125rem;
        }}

        .ens-stage-desc {{
            font-size: 0.75rem;
            color: var(--text-tertiary);
        }}

        .ens-stage-dot {{
            width: 7px;
            height: 7px;
            background: var(--text-tertiary);
            border-radius: 50%;
        }}

        /* ===== Hero / Welcome ===== */
        .ens-hero {{
            background: transparent;
            border: none;
            border-radius: var(--radius-xl);
            padding: 1.2rem 1.2rem 0.65rem;
            margin: 0 auto;
            max-width: 980px;
            text-align: center !important;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: none;
        }}

        .ens-hero-icon {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}

        .ens-hero-title {{
            font-family: var(--font-primary);
            font-size: 1.35rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }}

        .ens-hero-subtitle {{
            font-size: 0.875rem;
            color: var(--text-tertiary);
            width: 100%;
            max-width: 760px;
            margin: 0 auto 0.35rem auto;
            text-align: center !important;
            text-wrap: balance;
        }}

        /* ===== Chat Messages ===== */
        .stChatMessage {{
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 0.875rem 1rem !important;
            margin-bottom: 0.5rem !important;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        [data-testid="stAppViewContainer"] [data-testid="stMain"] {{
            scrollbar-gutter: stable;
        }}

        [data-testid="stAppViewContainer"] [data-testid="stMain"] .block-container {{
            overflow-anchor: none;
        }}

        .user-message-container {{
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.75rem;
        }}

        .user-message {{
            background: var(--accent);
            color: #FFFFFF;
            padding: 0.625rem 1rem;
            border-radius: 14px 14px 4px 14px;
            max-width: 72%;
            font-size: 0.875rem;
            line-height: 1.5;
            box-shadow: 0 1px 3px rgba(90, 54, 131, 0.2);
        }}

        .user-avatar {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-left: 0.5rem;
            flex-shrink: 0;
            font-size: 1rem;
        }}

        .assistant-message-container {{
            display: flex;
            justify-content: flex-start;
            margin-bottom: 0.75rem;
        }}

        .assistant-message {{
            background: var(--surface);
            color: var(--text-primary);
            padding: 0.625rem 1rem;
            border-radius: 14px 14px 14px 4px;
            max-width: 72%;
            font-size: 0.875rem;
            line-height: 1.5;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px var(--shadow);
        }}

        .assistant-avatar {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: var(--accent-light);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.5rem;
            flex-shrink: 0;
            font-size: 1rem;
        }}

        .blinking-cursor {{
            display: none !important;
        }}

        .ens-streaming-indicator {{
            color: var(--text-tertiary);
            font-size: 0.75rem;
            letter-spacing: 0.03em;
            white-space: nowrap;
        }}

        .ens-chat-surface {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            padding: 1.25rem;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        /* ===== Status Indicators ===== */
        .status-dot {{
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
        }}

        .status-dot.active {{
            background: var(--success);
            box-shadow: 0 0 4px var(--success);
            animation: pulse 2s infinite;
        }}

        .status-dot.idle {{
            background: var(--text-tertiary);
        }}

        .status-dot.error {{
            background: var(--error);
        }}

        /* ===== Provider Badge ===== */
        .ens-provider-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.2rem 0.5rem;
            border-radius: 999px;
            margin: 0.35rem 0;
            background: var(--bg-primary);
            border: 1px solid var(--border);
        }}

        .ens-provider-dot {{
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--text-tertiary);
        }}

        .ens-provider-label {{
            font-size: 0.6875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }}

        .ens-provider-badge[data-provider="azure"] {{
            background: rgba(0, 120, 212, 0.08);
            border-color: rgba(0, 120, 212, 0.15);
        }}
        .ens-provider-badge[data-provider="azure"] .ens-provider-dot {{ background: #0078D4; }}
        .ens-provider-badge[data-provider="azure"] .ens-provider-label {{ color: #0078D4; }}

        .ens-provider-badge[data-provider="openai"] {{
            background: rgba(16, 163, 127, 0.08);
            border-color: rgba(16, 163, 127, 0.15);
        }}
        .ens-provider-badge[data-provider="openai"] .ens-provider-dot {{ background: #10A37F; }}
        .ens-provider-badge[data-provider="openai"] .ens-provider-label {{ color: #10A37F; }}

        .ens-provider-badge[data-provider="openai_compatible"] {{
            background: rgba(16, 163, 127, 0.08);
            border-color: rgba(16, 163, 127, 0.15);
        }}
        .ens-provider-badge[data-provider="openai_compatible"] .ens-provider-dot {{ background: #10A37F; }}
        .ens-provider-badge[data-provider="openai_compatible"] .ens-provider-label {{ color: #10A37F; }}

        .ens-provider-badge[data-provider="anthropic"] {{
            background: rgba(217, 119, 87, 0.08);
            border-color: rgba(217, 119, 87, 0.15);
        }}
        .ens-provider-badge[data-provider="anthropic"] .ens-provider-dot {{ background: #D97757; }}
        .ens-provider-badge[data-provider="anthropic"] .ens-provider-label {{ color: #D97757; }}

        .ens-provider-badge[data-provider="others"] {{
            background: rgba(99, 102, 241, 0.08);
            border-color: rgba(99, 102, 241, 0.15);
        }}
        .ens-provider-badge[data-provider="others"] .ens-provider-dot {{ background: #6366F1; }}
        .ens-provider-badge[data-provider="others"] .ens-provider-label {{ color: #6366F1; }}

        .ens-provider-badge[data-provider="openrouter"],
        .ens-provider-badge[data-provider="deepseek"],
        .ens-provider-badge[data-provider="groq"],
        .ens-provider-badge[data-provider="together_ai"],
        .ens-provider-badge[data-provider="mistral"],
        .ens-provider-badge[data-provider="cohere"],
        .ens-provider-badge[data-provider="xai"],
        .ens-provider-badge[data-provider="perplexity"],
        .ens-provider-badge[data-provider="gemini"] {{
            background: rgba(37, 99, 235, 0.08);
            border-color: rgba(37, 99, 235, 0.16);
        }}
        .ens-provider-badge[data-provider="openrouter"] .ens-provider-dot,
        .ens-provider-badge[data-provider="deepseek"] .ens-provider-dot,
        .ens-provider-badge[data-provider="groq"] .ens-provider-dot,
        .ens-provider-badge[data-provider="together_ai"] .ens-provider-dot,
        .ens-provider-badge[data-provider="mistral"] .ens-provider-dot,
        .ens-provider-badge[data-provider="cohere"] .ens-provider-dot,
        .ens-provider-badge[data-provider="xai"] .ens-provider-dot,
        .ens-provider-badge[data-provider="perplexity"] .ens-provider-dot,
        .ens-provider-badge[data-provider="gemini"] .ens-provider-dot {{
            background: #2563EB;
        }}
        .ens-provider-badge[data-provider="openrouter"] .ens-provider-label,
        .ens-provider-badge[data-provider="deepseek"] .ens-provider-label,
        .ens-provider-badge[data-provider="groq"] .ens-provider-label,
        .ens-provider-badge[data-provider="together_ai"] .ens-provider-label,
        .ens-provider-badge[data-provider="mistral"] .ens-provider-label,
        .ens-provider-badge[data-provider="cohere"] .ens-provider-label,
        .ens-provider-badge[data-provider="xai"] .ens-provider-label,
        .ens-provider-badge[data-provider="perplexity"] .ens-provider-label,
        .ens-provider-badge[data-provider="gemini"] .ens-provider-label {{
            color: #2563EB;
        }}

        /* ===== Agent Cards ===== */
        .agent-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 1px 3px var(--shadow);
            transition: box-shadow 0.15s ease;
        }}

        .agent-card:hover {{
            box-shadow: 0 2px 8px var(--shadow-hover);
        }}

        .agent-card .agent-icon {{
            width: 36px;
            height: 36px;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        .agent-card .agent-info h4 {{
            margin: 0;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .agent-card .agent-info p {{
            margin: 0;
            font-size: 0.75rem;
            color: var(--text-tertiary);
        }}

        .ens-agent-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            box-shadow: 0 1px 3px var(--shadow);
            margin-bottom: 0.75rem;
            transition: box-shadow 0.15s ease;
        }}

        .ens-agent-card[data-status="active"] {{
            border-left: 3px solid var(--accent);
        }}

        .ens-agent-card[data-status="error"] {{
            border-left: 3px solid var(--error);
        }}

        .ens-agent-icon {{
            width: 36px;
            height: 36px;
            background: var(--bg-primary);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        .ens-agent-meta {{
            display: flex;
            flex-direction: column;
            gap: 0.125rem;
            flex: 1;
        }}

        .ens-agent-title {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .ens-agent-desc {{
            font-size: 0.75rem;
            color: var(--text-tertiary);
        }}

        .ens-agent-status {{
            font-size: 0.625rem;
            font-weight: 600;
            color: var(--text-tertiary);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-left: auto;
        }}

        .ens-agent-progress {{
            height: 3px;
            background: var(--bg-secondary);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.5rem;
            width: 100%;
        }}

        .ens-agent-progress-bar {{
            height: 100%;
            background: var(--accent);
            border-radius: 999px;
            transition: width 0.3s ease;
            width: var(--progress, 0%);
        }}

        /* ===== Activity Log ===== */
        .ens-log-row {{
            display: flex;
            gap: 0.75rem;
            padding: 0.4rem 0;
            border-bottom: 1px solid var(--border);
            font-size: 0.75rem;
        }}

        .ens-log-time {{
            color: var(--text-tertiary);
            font-family: var(--font-mono);
            font-size: 0.6875rem;
            flex-shrink: 0;
        }}

        .ens-log-message {{
            color: var(--text-secondary);
        }}

        .ens-log-message[data-level="info"] {{ color: var(--text-tertiary); }}
        .ens-log-message[data-level="success"] {{ color: var(--success); }}
        .ens-log-message[data-level="warning"] {{ color: var(--warning); }}
        .ens-log-message[data-level="error"] {{ color: var(--error); }}

        /* ===== Reasoning / Thinking ===== */
        .reasoning-step {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-secondary);
            box-shadow: 0 1px 3px var(--shadow);
        }}

        .reasoning-step .step-header {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .ens-reasoning-panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
        }}

        .ens-reasoning-panel[data-live="true"] {{
            border-left-color: var(--accent);
        }}

        .ens-reasoning-step {{
            margin-bottom: 0.5rem;
        }}

        .ens-reasoning-step-header {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.2rem;
        }}

        .ens-reasoning-step-tag {{
            background: var(--bg-primary);
            color: var(--text-secondary);
            font-size: 0.625rem;
            font-weight: 600;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
        }}

        .ens-reasoning-step-title {{
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .ens-reasoning-step-duration {{
            font-size: 0.6875rem;
            color: var(--text-tertiary);
        }}

        .ens-reasoning-step-body {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            padding-left: 0.25rem;
            font-family: var(--font-mono);
        }}

        .ens-reasoning-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 2px solid var(--border);
            padding: 0.625rem 0.75rem;
            margin-bottom: 0.35rem;
            border-radius: var(--radius-sm);
        }}

        .ens-reasoning-card-title {{
            font-size: 0.6875rem;
            color: var(--text-tertiary);
            margin-bottom: 0.15rem;
        }}

        .ens-reasoning-card-body {{
            font-size: 0.75rem;
            color: var(--text-primary);
            font-family: var(--font-mono);
            white-space: pre-wrap;
        }}

        /* ===== Thinking Banner ===== */
        .ens-thinking-banner {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.75rem 1rem;
            background: #F3EFF8;
            border: 1px solid rgba(90, 54, 131, 0.15);
            border-radius: var(--radius-md);
            margin-bottom: 0.75rem;
        }}

        .ens-thinking-text {{
            font-size: 0.8125rem;
            color: var(--accent);
            font-weight: 500;
        }}

        /* ===== Alert ===== */
        .ens-alert {{
            background: #FFFCF5;
            border: 1px solid #F0E4C8;
            border-radius: var(--radius-lg);
            padding: 0.875rem 1.125rem;
            margin-bottom: 0.75rem;
        }}

        .ens-alert-header {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.3rem;
        }}

        .ens-alert-icon {{
            font-size: 0.875rem;
        }}

        .ens-alert-title {{
            font-size: 0.8125rem;
            font-weight: 600;
            color: #7A5B00;
        }}

        .ens-alert-text {{
            font-size: 0.75rem;
            color: #7A5B00;
            margin: 0;
        }}

        /* ===== Empty States ===== */
        .ens-empty-state {{
            text-align: center;
            color: var(--text-tertiary);
        }}

        .ens-empty-state-sm {{
            padding: 1.5rem;
            font-size: 0.8125rem;
        }}

        .ens-empty-state-lg {{
            padding: 2.5rem;
            font-size: 0.875rem;
        }}

        .ens-empty-icon {{
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }}

        .ens-empty-text {{
            margin: 0;
        }}

        .ens-muted-text {{
            font-size: 0.75rem;
            color: var(--text-tertiary);
        }}

        .ens-muted-italic {{
            font-style: italic;
        }}

        /* ===== Expander ===== */
        div[data-testid="stExpander"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        div[data-testid="stExpander"] > div:first-child {{
            background: transparent;
            border: none;
        }}

        /* ===== Progress Bar ===== */
        .stProgress > div > div {{
            background: var(--bg-secondary) !important;
            border-radius: 999px !important;
        }}

        .stProgress > div > div > div {{
            background: var(--accent) !important;
            border-radius: 999px !important;
        }}

        /* ===== Spacers ===== */
        .ens-spacer-sm {{
            height: 0.75rem;
        }}

        .ens-spacer-md {{
            height: 1.25rem;
        }}

        /* ===== Settings Page Cards ===== */
        .ens-settings-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px var(--shadow);
        }}

        .ens-settings-card-title {{
            font-size: 0.9375rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.125rem;
        }}

        .ens-settings-card-desc {{
            font-size: 0.75rem;
            color: var(--text-tertiary);
            margin-bottom: 0.875rem;
        }}

        /* ===== Animations ===== */
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        @keyframes blink {{
            0%, 50% {{ opacity: 1; }}
            51%, 100% {{ opacity: 0; }}
        }}

        .thinking-pulse {{
            display: flex;
            gap: 3px;
            padding: 0.4rem 0;
        }}

        .thinking-pulse span {{
            width: 5px;
            height: 5px;
            background: var(--text-tertiary);
            border-radius: 50%;
            animation: thinking 1.4s infinite ease-in-out both;
        }}

        .thinking-pulse span:nth-child(1) {{ animation-delay: -0.32s; }}
        .thinking-pulse span:nth-child(2) {{ animation-delay: -0.16s; }}
        .thinking-pulse span:nth-child(3) {{ animation-delay: 0s; }}

        @keyframes thinking {{
            0%, 80%, 100% {{
                transform: scale(0.8);
                opacity: 0.5;
            }}
            40% {{
                transform: scale(1);
                opacity: 1;
            }}
        }}

        /* ===== Scrollbar ===== */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}

        ::-webkit-scrollbar-track {{
            background: transparent;
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border);
            border-radius: 3px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-tertiary);
        }}

        /* ===== Tooltip ===== */
        .stTooltipIcon {{
            color: var(--text-tertiary) !important;
        }}

        /* ===== Data Table ===== */
        .stDataFrame {{
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-lg) !important;
            overflow: hidden;
        }}

        /* ===== File Uploader ===== */
        .stFileUploader > div {{
            background: var(--surface) !important;
            border: 2px dashed var(--border) !important;
            border-radius: var(--radius-lg) !important;
            padding: 1.5rem !important;
            transition: border-color 0.15s ease !important;
        }}

        .stFileUploader > div:hover {{
            border-color: var(--accent) !important;
        }}

        /* ===== Conversation History (Sidebar) ===== */
        .ens-history-label {{
            font-family: {FONTS["primary"]};
            font-size: 0.58rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-tertiary);
            padding: 0.46rem 0.15rem 0.24rem;
        }}

        .ens-history-empty {{
            font-family: {FONTS["primary"]};
            font-size: 0.60rem;
            color: var(--text-tertiary);
            text-align: center;
            padding: 1rem 0;
        }}

        .ens-sidebar-divider {{
            height: 1px;
            width: 100%;
            background: var(--border);
            margin: 0.65rem 0 0.75rem;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-new-chat-anchor) .stButton > button {{
            color: #000000 !important;
            font-size: 0.66rem !important;
            font-weight: 500 !important;
            min-height: 1.5rem !important;
            padding: 0.2rem 0.44rem !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) .stButton > button {{
            font-size: 0.44rem !important;
            padding: 0.18rem 0.34rem !important;
            min-height: 1.42rem !important;
            line-height: 1.15 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] {{
            border: 0 !important;
            outline: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] > button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] > div > button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"] {{
            min-height: 1.2rem !important;
            min-width: 0.9rem !important;
            padding: 0.1rem 0.05rem !important;
            justify-content: center !important;
            font-size: 0 !important;
            font-weight: 600 !important;
            line-height: 1 !important;
            color: #111111 !important;
            border: 0 !important;
            border-width: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            outline: none !important;
            outline-width: 0 !important;
            outline-offset: 0 !important;
            border-color: transparent !important;
            -webkit-appearance: none !important;
            appearance: none !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] > button::before,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] > button::after,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]::before,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]::after {{
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            content: none !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button[aria-expanded="true"],
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"],
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"][aria-expanded="true"] {{
            border: none !important;
            border-width: 0 !important;
            border-color: transparent !important;
            outline: none !important;
            outline-width: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button:hover,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button:focus,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button:focus-visible,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button:active,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button:hover,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button:focus,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button:focus-visible,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button:active,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]:hover,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]:focus,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]:focus-visible,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"]:active {{
            background: transparent !important;
            border: none !important;
            border-width: 0 !important;
            box-shadow: none !important;
            outline: none !important;
            outline-width: 0 !important;
            color: #111111 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopover"] button svg,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-baseweb="popover"] > button svg,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) button[aria-haspopup="dialog"] svg {{
            color: #111111 !important;
            fill: #111111 !important;
            stroke: #111111 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stButton > button,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stDownloadButton > button {{
            font-size: 0.52rem !important;
            min-height: 1.42rem !important;
            padding: 0.16rem 0.3rem !important;
            color: var(--text-secondary) !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stButton > button:hover,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stButton > button:focus,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stDownloadButton > button:hover,
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]:has(.ens-history-list-anchor) [data-testid="stPopoverPopover"] .stDownloadButton > button:focus {{
            color: var(--text-primary) !important;
            background: var(--bg-primary) !important;
            border: 1px solid var(--border) !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        .ens-chat-empty-anchor {{
            display: none;
        }}

        div[data-testid="stVerticalBlock"]:has(.ens-chat-empty-anchor) {{
            margin-top: 0 !important;
            min-height: calc(100svh - 210px);
            padding-top: clamp(1rem, 4svh, 2.6rem);
        }}

        @media (max-width: 768px) {{
            div[data-testid="stVerticalBlock"]:has(.ens-chip-actions-anchor) {{
                margin-top: 0.2rem;
            }}

            div[data-testid="stVerticalBlock"]:has(.ens-chat-empty-anchor) {{
                min-height: calc(100svh - 184px);
                padding-top: clamp(0.75rem, 2.6svh, 1.8rem);
            }}
        }}

        /* ===== Responsive ===== */
        @media (max-width: 768px) {{
            h1 {{
                font-size: 1.35rem !important;
            }}
        }}
    </style>
    """


def get_thinking_animation() -> str:
    """Return HTML for the thinking animation."""
    return """
    <div class="thinking-pulse">
        <span></span>
        <span></span>
        <span></span>
    </div>
    """


def get_agent_card(icon: str, name: str, status: str, description: str) -> str:
    """Return HTML for an agent status card."""
    status_class = "active" if status == "active" else ("error" if status == "error" else "idle")
    safe_icon = html.escape(str(icon), quote=False)
    safe_name = html.escape(str(name), quote=False)
    safe_desc = html.escape(str(description), quote=False)
    return f"""
    <div class="agent-card">
        <div class="agent-icon">{safe_icon}</div>
        <div class="agent-info">
            <h4><span class="status-dot {status_class}"></span>{safe_name}</h4>
            <p>{safe_desc}</p>
        </div>
    </div>
    """


def get_reasoning_step(step_num: int, title: str, content: str) -> str:
    """Return HTML for a reasoning step in the thinking log."""
    safe_title = html.escape(str(title), quote=False)
    safe_content = html.escape(str(content), quote=False).replace("\n", "<br/>")
    return f"""
    <div class="reasoning-step">
        <div class="step-header">
            <span>Step {step_num}</span>
            <span>&middot;</span>
            <span>{safe_title}</span>
        </div>
        <div class="step-content">{safe_content}</div>
    </div>
    """
