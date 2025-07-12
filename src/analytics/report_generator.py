"""
Report Generator Module

Generates comprehensive performance reports in multiple formats including
PDF, HTML, Excel, and JSON with customizable templates and content.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
from pathlib import Path
import base64
from io import BytesIO

from ..core.interfaces import Injectable
from ..core.decorators import injectable
from ..core.logger import get_logger
from .performance_analyzer import PerformanceAnalyzer, AnalysisTimeframe

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Available report formats"""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportSection(Enum):
    """Report sections"""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_OVERVIEW = "performance_overview"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    STRATEGY_BREAKDOWN = "strategy_breakdown"
    RISK_ANALYSIS = "risk_analysis"
    TRADE_ANALYSIS = "trade_analysis"
    MARKET_ANALYSIS = "market_analysis"
    RECOMMENDATIONS = "recommendations"


@dataclass
class ReportConfig:
    """Report generation configuration"""
    title: str = "Trading Performance Report"
    subtitle: str = ""
    author: str = "Quantum Trading Platform"
    include_sections: List[ReportSection] = None
    exclude_sections: List[ReportSection] = None
    include_charts: bool = True
    include_tables: bool = True
    include_metrics: bool = True
    chart_style: str = "dark"
    logo_path: Optional[str] = None
    custom_css: Optional[str] = None
    template_path: Optional[str] = None
    
    def __post_init__(self):
        if self.include_sections is None:
            self.include_sections = list(ReportSection)


@dataclass
class ReportContent:
    """Report content structure"""
    metadata: Dict[str, Any]
    sections: Dict[str, Dict[str, Any]]
    charts: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    metrics: Dict[str, Any]


@injectable
class ReportGenerator(Injectable):
    """Generates performance reports in various formats"""
    
    def __init__(
        self,
        performance_analyzer: PerformanceAnalyzer = None,
        template_dir: Optional[Path] = None
    ):
        self.performance_analyzer = performance_analyzer
        self.template_dir = template_dir or Path(__file__).parent / "templates"
        
        # Report templates
        self._templates: Dict[ReportFormat, str] = {}
        self._load_templates()
        
        # Chart generators
        self._chart_generators: Dict[str, callable] = {
            "equity_curve": self._generate_equity_curve_chart,
            "returns_distribution": self._generate_returns_distribution_chart,
            "drawdown_chart": self._generate_drawdown_chart,
            "monthly_returns": self._generate_monthly_returns_heatmap,
            "strategy_comparison": self._generate_strategy_comparison_chart,
            "risk_metrics": self._generate_risk_metrics_chart
        }
    
    async def generate_report(
        self,
        format: ReportFormat,
        config: ReportConfig = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Generate performance report
        
        Args:
            format: Output format
            config: Report configuration
            start_date: Report period start
            end_date: Report period end
            output_path: Optional output file path
            
        Returns:
            Report content as string or bytes
        """
        config = config or ReportConfig()
        
        # Get performance analysis
        analysis = await self.performance_analyzer.analyze(
            start_date=start_date,
            end_date=end_date,
            timeframe=AnalysisTimeframe.DAILY
        )
        
        # Build report content
        content = await self._build_report_content(analysis, config, start_date, end_date)
        
        # Generate report in requested format
        if format == ReportFormat.PDF:
            report_data = await self._generate_pdf_report(content, config)
        elif format == ReportFormat.HTML:
            report_data = await self._generate_html_report(content, config)
        elif format == ReportFormat.EXCEL:
            report_data = await self._generate_excel_report(content, config)
        elif format == ReportFormat.JSON:
            report_data = await self._generate_json_report(content, config)
        elif format == ReportFormat.MARKDOWN:
            report_data = await self._generate_markdown_report(content, config)
        elif format == ReportFormat.CSV:
            report_data = await self._generate_csv_report(content, config)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if path provided
        if output_path:
            await self._save_report(report_data, output_path, format)
        
        return report_data
    
    async def generate_custom_report(
        self,
        template: str,
        data: Dict[str, Any],
        format: ReportFormat = ReportFormat.HTML
    ) -> str:
        """Generate report from custom template"""
        # TODO: Implement custom template rendering
        pass
    
    async def _build_report_content(
        self,
        analysis: Dict[str, Any],
        config: ReportConfig,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> ReportContent:
        """Build report content from analysis"""
        # Build metadata
        metadata = {
            "title": config.title,
            "subtitle": config.subtitle,
            "author": config.author,
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else "Beginning",
                "end": end_date.isoformat() if end_date else "Present"
            }
        }
        
        # Build sections
        sections = {}
        
        if ReportSection.EXECUTIVE_SUMMARY in config.include_sections:
            sections["executive_summary"] = await self._build_executive_summary(analysis)
        
        if ReportSection.PERFORMANCE_OVERVIEW in config.include_sections:
            sections["performance_overview"] = await self._build_performance_overview(analysis)
        
        if ReportSection.PORTFOLIO_ANALYSIS in config.include_sections:
            sections["portfolio_analysis"] = await self._build_portfolio_analysis(analysis)
        
        if ReportSection.STRATEGY_BREAKDOWN in config.include_sections:
            sections["strategy_breakdown"] = await self._build_strategy_breakdown(analysis)
        
        if ReportSection.RISK_ANALYSIS in config.include_sections:
            sections["risk_analysis"] = await self._build_risk_analysis(analysis)
        
        if ReportSection.TRADE_ANALYSIS in config.include_sections:
            sections["trade_analysis"] = await self._build_trade_analysis(analysis)
        
        if ReportSection.MARKET_ANALYSIS in config.include_sections:
            sections["market_analysis"] = await self._build_market_analysis(analysis)
        
        if ReportSection.RECOMMENDATIONS in config.include_sections:
            sections["recommendations"] = await self._build_recommendations(analysis)
        
        # Generate charts if enabled
        charts = {}
        if config.include_charts:
            charts = await self._generate_charts(analysis)
        
        # Generate tables if enabled
        tables = {}
        if config.include_tables:
            tables = await self._generate_tables(analysis)
        
        # Extract key metrics
        metrics = {}
        if config.include_metrics:
            metrics = self._extract_key_metrics(analysis)
        
        return ReportContent(
            metadata=metadata,
            sections=sections,
            charts=charts,
            tables=tables,
            metrics=metrics
        )
    
    async def _build_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section"""
        summary = analysis.get("summary", {})
        
        return {
            "title": "Executive Summary",
            "content": {
                "overview": f"The trading system generated a total return of {summary.get('total_return', 0):.2%} "
                           f"with a Sharpe ratio of {summary.get('sharpe_ratio', 0):.2f}.",
                "key_metrics": {
                    "Total Return": f"{summary.get('total_return', 0):.2%}",
                    "Total P&L": f"${summary.get('total_pnl', 0):,.2f}",
                    "Win Rate": f"{summary.get('win_rate', 0):.1%}",
                    "Max Drawdown": f"{summary.get('max_drawdown', 0):.2%}",
                    "Sharpe Ratio": f"{summary.get('sharpe_ratio', 0):.2f}",
                    "Profit Factor": f"{summary.get('profit_factor', 0):.2f}"
                },
                "highlights": [
                    f"Executed {summary.get('trades_count', 0)} trades",
                    f"Average trade P&L: ${summary.get('avg_trade', 0):.2f}",
                    f"Best trade: ${summary.get('best_trade', 0):.2f}",
                    f"Worst trade: ${summary.get('worst_trade', 0):.2f}"
                ]
            }
        }
    
    async def _build_performance_overview(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build performance overview section"""
        portfolio = analysis.get("portfolio", {})
        
        return {
            "title": "Performance Overview",
            "content": {
                "returns_analysis": portfolio.get("returns_analysis", {}),
                "risk_metrics": portfolio.get("risk_metrics", {}),
                "efficiency_metrics": portfolio.get("efficiency_metrics", {})
            }
        }
    
    async def _build_portfolio_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build portfolio analysis section"""
        portfolio = analysis.get("portfolio", {})
        
        return {
            "title": "Portfolio Analysis",
            "content": {
                "composition": portfolio.get("composition", {}),
                "allocation": portfolio.get("allocation", {}),
                "concentration": portfolio.get("concentration", {}),
                "correlation": portfolio.get("correlation", {})
            }
        }
    
    async def _build_strategy_breakdown(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build strategy breakdown section"""
        strategies = analysis.get("strategies", {})
        
        strategy_summaries = []
        for strategy_id, strategy in strategies.items():
            summary = {
                "name": strategy.strategy_name,
                "metrics": {
                    "Total Return": f"{strategy.metrics.total_return:.2%}",
                    "Sharpe Ratio": f"{strategy.metrics.sharpe_ratio:.2f}",
                    "Win Rate": f"{strategy.metrics.win_rate:.1%}",
                    "Trades": strategy.metrics.total_trades
                }
            }
            strategy_summaries.append(summary)
        
        return {
            "title": "Strategy Performance Breakdown",
            "content": {
                "strategies": strategy_summaries,
                "comparison": analysis.get("strategy_comparison", {})
            }
        }
    
    async def _build_risk_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk analysis section"""
        risk = analysis.get("risk_analysis", {})
        
        return {
            "title": "Risk Analysis",
            "content": {
                "var_analysis": risk.get("value_at_risk", {}),
                "stress_testing": risk.get("stress_testing", {}),
                "drawdown_analysis": risk.get("drawdown_analysis", {}),
                "exposure_analysis": risk.get("exposure_analysis", {})
            }
        }
    
    async def _build_trade_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build trade analysis section"""
        summary = analysis.get("summary", {})
        
        return {
            "title": "Trade Analysis",
            "content": {
                "statistics": {
                    "Total Trades": summary.get("trades_count", 0),
                    "Winning Trades": int(summary.get("trades_count", 0) * summary.get("win_rate", 0)),
                    "Losing Trades": int(summary.get("trades_count", 0) * (1 - summary.get("win_rate", 0))),
                    "Average Win": f"${summary.get('avg_win', 0):.2f}",
                    "Average Loss": f"${summary.get('avg_loss', 0):.2f}",
                    "Profit Factor": f"{summary.get('profit_factor', 0):.2f}"
                },
                "distribution": analysis.get("trade_distribution", {}),
                "time_analysis": analysis.get("time_patterns", {})
            }
        }
    
    async def _build_market_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build market analysis section"""
        return {
            "title": "Market Analysis",
            "content": {
                "symbol_performance": analysis.get("symbols", {}),
                "exchange_performance": analysis.get("exchanges", {}),
                "market_regimes": analysis.get("market_regimes", {}),
                "correlation_analysis": analysis.get("correlation", {})
            }
        }
    
    async def _build_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section"""
        recommendations = []
        
        # Generate recommendations based on analysis
        summary = analysis.get("summary", {})
        
        if summary.get("win_rate", 0) < 0.4:
            recommendations.append({
                "category": "Strategy",
                "priority": "High",
                "recommendation": "Consider reviewing and optimizing strategy entry criteria to improve win rate"
            })
        
        if summary.get("max_drawdown", 0) > 0.2:
            recommendations.append({
                "category": "Risk",
                "priority": "High",
                "recommendation": "Implement stricter risk management to reduce maximum drawdown"
            })
        
        if summary.get("sharpe_ratio", 0) < 1.0:
            recommendations.append({
                "category": "Performance",
                "priority": "Medium",
                "recommendation": "Focus on improving risk-adjusted returns through better position sizing"
            })
        
        return {
            "title": "Recommendations",
            "content": {
                "recommendations": recommendations,
                "action_items": self._generate_action_items(analysis)
            }
        }
    
    async def _generate_charts(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all charts for the report"""
        charts = {}
        
        for chart_name, generator in self._chart_generators.items():
            try:
                charts[chart_name] = await generator(analysis)
            except Exception as e:
                logger.error(f"Error generating {chart_name} chart: {e}")
        
        return charts
    
    async def _generate_tables(self, analysis: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Generate all tables for the report"""
        tables = {}
        
        # Strategy performance table
        if "strategies" in analysis:
            strategy_data = []
            for sid, strategy in analysis["strategies"].items():
                strategy_data.append({
                    "Strategy": strategy.strategy_name,
                    "Total Return": strategy.metrics.total_return,
                    "Sharpe Ratio": strategy.metrics.sharpe_ratio,
                    "Max Drawdown": strategy.metrics.max_drawdown,
                    "Win Rate": strategy.metrics.win_rate,
                    "Trades": strategy.metrics.total_trades
                })
            tables["strategy_performance"] = pd.DataFrame(strategy_data)
        
        # Symbol performance table
        if "symbols" in analysis:
            symbol_data = []
            for symbol, data in analysis["symbols"].items():
                symbol_data.append({
                    "Symbol": symbol,
                    "Trades": data["trades_count"],
                    "Total Volume": data["total_volume"],
                    "Avg Trade Size": data["avg_trade_size"],
                    "Avg Holding Time": data["avg_holding_time"]
                })
            tables["symbol_performance"] = pd.DataFrame(symbol_data)
        
        return tables
    
    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis"""
        summary = analysis.get("summary", {})
        
        return {
            "performance": {
                "total_return": summary.get("total_return", 0),
                "annualized_return": summary.get("annualized_return", 0),
                "volatility": summary.get("volatility", 0),
                "sharpe_ratio": summary.get("sharpe_ratio", 0),
                "sortino_ratio": summary.get("sortino_ratio", 0),
                "calmar_ratio": summary.get("calmar_ratio", 0)
            },
            "risk": {
                "max_drawdown": summary.get("max_drawdown", 0),
                "value_at_risk": summary.get("value_at_risk", 0),
                "conditional_var": summary.get("conditional_var", 0)
            },
            "trading": {
                "total_trades": summary.get("trades_count", 0),
                "win_rate": summary.get("win_rate", 0),
                "profit_factor": summary.get("profit_factor", 0),
                "avg_trade": summary.get("avg_trade", 0)
            }
        }
    
    def _generate_action_items(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate action items based on analysis"""
        action_items = []
        
        # Add action items based on performance
        summary = analysis.get("summary", {})
        
        if summary.get("trades_count", 0) < 100:
            action_items.append({
                "action": "Increase sample size",
                "description": "Consider running the system longer to gather more trade data"
            })
        
        return action_items
    
    async def _generate_equity_curve_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate equity curve chart data"""
        # TODO: Implement chart generation
        return {
            "type": "line",
            "data": {},
            "options": {}
        }
    
    async def _generate_returns_distribution_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate returns distribution chart"""
        return {
            "type": "histogram",
            "data": {},
            "options": {}
        }
    
    async def _generate_drawdown_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate drawdown chart"""
        return {
            "type": "area",
            "data": {},
            "options": {}
        }
    
    async def _generate_monthly_returns_heatmap(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monthly returns heatmap"""
        return {
            "type": "heatmap",
            "data": {},
            "options": {}
        }
    
    async def _generate_strategy_comparison_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy comparison chart"""
        return {
            "type": "bar",
            "data": {},
            "options": {}
        }
    
    async def _generate_risk_metrics_chart(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk metrics chart"""
        return {
            "type": "radar",
            "data": {},
            "options": {}
        }
    
    async def _generate_pdf_report(self, content: ReportContent, config: ReportConfig) -> bytes:
        """Generate PDF report"""
        # TODO: Implement PDF generation using reportlab or similar
        return b"PDF content"
    
    async def _generate_html_report(self, content: ReportContent, config: ReportConfig) -> str:
        """Generate HTML report"""
        template = self._templates.get(ReportFormat.HTML, self._get_default_html_template())
        
        # Render template with content
        html = template.format(
            title=content.metadata["title"],
            subtitle=content.metadata["subtitle"],
            author=content.metadata["author"],
            generated_at=content.metadata["generated_at"],
            sections=self._render_html_sections(content.sections),
            charts=self._render_html_charts(content.charts),
            tables=self._render_html_tables(content.tables),
            custom_css=config.custom_css or ""
        )
        
        return html
    
    async def _generate_excel_report(self, content: ReportContent, config: ReportConfig) -> bytes:
        """Generate Excel report"""
        # Create Excel writer
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write summary sheet
            summary_df = pd.DataFrame([content.metrics["performance"]])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write tables
            for table_name, df in content.tables.items():
                df.to_excel(writer, sheet_name=table_name[:31], index=False)
        
        output.seek(0)
        return output.read()
    
    async def _generate_json_report(self, content: ReportContent, config: ReportConfig) -> str:
        """Generate JSON report"""
        return json.dumps({
            "metadata": content.metadata,
            "sections": content.sections,
            "metrics": content.metrics,
            "tables": {name: df.to_dict() for name, df in content.tables.items()}
        }, indent=2)
    
    async def _generate_markdown_report(self, content: ReportContent, config: ReportConfig) -> str:
        """Generate Markdown report"""
        md_lines = []
        
        # Title
        md_lines.append(f"# {content.metadata['title']}")
        if content.metadata['subtitle']:
            md_lines.append(f"## {content.metadata['subtitle']}")
        md_lines.append("")
        
        # Metadata
        md_lines.append(f"**Generated:** {content.metadata['generated_at']}")
        md_lines.append(f"**Period:** {content.metadata['period']['start']} to {content.metadata['period']['end']}")
        md_lines.append("")
        
        # Sections
        for section_name, section in content.sections.items():
            md_lines.append(f"## {section['title']}")
            md_lines.append("")
            
            # Render section content
            self._render_markdown_content(section['content'], md_lines)
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    async def _generate_csv_report(self, content: ReportContent, config: ReportConfig) -> str:
        """Generate CSV report (metrics only)"""
        # Flatten metrics into CSV format
        rows = []
        
        for category, metrics in content.metrics.items():
            for metric_name, value in metrics.items():
                rows.append({
                    "Category": category,
                    "Metric": metric_name,
                    "Value": value
                })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    def _render_html_sections(self, sections: Dict[str, Dict[str, Any]]) -> str:
        """Render sections as HTML"""
        html_parts = []
        
        for section_name, section in sections.items():
            html_parts.append(f'<section id="{section_name}">')
            html_parts.append(f'<h2>{section["title"]}</h2>')
            html_parts.append(self._render_html_content(section["content"]))
            html_parts.append('</section>')
        
        return "\n".join(html_parts)
    
    def _render_html_content(self, content: Dict[str, Any]) -> str:
        """Render content as HTML"""
        # TODO: Implement proper HTML rendering
        return f"<pre>{json.dumps(content, indent=2)}</pre>"
    
    def _render_html_charts(self, charts: Dict[str, Any]) -> str:
        """Render charts as HTML"""
        # TODO: Implement chart rendering with Chart.js or similar
        return ""
    
    def _render_html_tables(self, tables: Dict[str, pd.DataFrame]) -> str:
        """Render tables as HTML"""
        html_parts = []
        
        for table_name, df in tables.items():
            html_parts.append(f'<h3>{table_name.replace("_", " ").title()}</h3>')
            html_parts.append(df.to_html(classes="table table-striped", index=False))
        
        return "\n".join(html_parts)
    
    def _render_markdown_content(self, content: Any, lines: List[str], indent: int = 0):
        """Render content as Markdown"""
        prefix = "  " * indent
        
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}**{key}:**")
                    self._render_markdown_content(value, lines, indent + 1)
                else:
                    lines.append(f"{prefix}- **{key}:** {value}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    self._render_markdown_content(item, lines, indent)
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{content}")
    
    async def _save_report(self, report_data: Union[str, bytes], output_path: str, format: ReportFormat):
        """Save report to file"""
        path = Path(output_path)
        
        if isinstance(report_data, str):
            path.write_text(report_data)
        else:
            path.write_bytes(report_data)
        
        logger.info(f"Report saved to {output_path}")
    
    def _load_templates(self):
        """Load report templates"""
        # Load default templates
        self._templates[ReportFormat.HTML] = self._get_default_html_template()
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .metric-value {{ font-size: 1.2em; color: #2196F3; }}
        {custom_css}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <h2>{subtitle}</h2>
    <p><strong>Author:</strong> {author}</p>
    <p><strong>Generated:</strong> {generated_at}</p>
    
    {sections}
    
    <div id="charts">
        {charts}
    </div>
    
    <div id="tables">
        {tables}
    </div>
</body>
</html>
"""