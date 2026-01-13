"""
Generate English HTML Report from Evaluation Results
"""

import json
import sys

# Set encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class EnglishReportGenerator:
    """English HTML Report Generator"""

    def __init__(self):
        """Initialize report generator"""
        self.template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Summarization & Reward Modeling Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .section h3 {{
            color: #764ba2;
            font-size: 1.3em;
            margin-bottom: 15px;
            margin-top: 25px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .info-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}

        .info-card .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}

        .info-card .value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
        }}

        .metric-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        }}

        .metric-card .metric-name {{
            font-size: 0.95em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .metric-card .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}

        .metric-card.excellent .metric-value {{
            color: #10b981;
        }}

        .metric-card.good .metric-value {{
            color: #3b82f6;
        }}

        .metric-card.average .metric-value {{
            color: #f59e0b;
        }}

        .paper-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 4px solid #764ba2;
        }}

        .paper-card h4 {{
            color: #333;
            font-size: 1.2em;
            margin-bottom: 15px;
        }}

        .paper-card .summary-box {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 3px solid #ddd;
        }}

        .paper-box .summary-box.generated {{
            border-left-color: #667eea;
        }}

        .paper-box .summary-box.reference {{
            border-left-color: #10b981;
        }}

        .paper-box .summary-label {{
            font-weight: bold;
            color: #666;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}

        .metrics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}

        .metric-badge {{
            display: inline-block;
            padding: 8px 16px;
            background: #e0e7ff;
            color: #4338ca;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.5s ease;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Evaluation Report: Multimodal Summarization</h1>
            <div class="subtitle">Reward Modeling for Academic Paper Summarization</div>
        </div>

        <div class="content">
            <!-- Basic Information -->
            <div class="section">
                <h2>Evaluation Information</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <div class="label">Evaluation Time</div>
                        <div class="value">{timestamp}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Total Papers</div>
                        <div class="value">{total_papers}</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Reward Model</div>
                        <div class="value">DeBERTa-v3</div>
                    </div>
                    <div class="info-card">
                        <div class="label">Generator Model</div>
                        <div class="value">Qwen3:8b</div>
                    </div>
                </div>
            </div>

            <!-- Summary Metrics -->
            <div class="section">
                <h2>Summary Metrics</h2>

                <h3>Reward Model Score</h3>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-name">Average Reward Score</div>
                        <div class="metric-value">{avg_reward_score:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {reward_progress}%"></div>
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-name">Standard Deviation</div>
                        <div class="metric-value">{std_reward_score:.4f}</div>
                    </div>
                </div>

                <h3>ROUGE Scores</h3>
                <div class="metrics-grid">
                    <div class="metric-card {rouge1_class}">
                        <div class="metric-name">ROUGE-1</div>
                        <div class="metric-value">{avg_rouge1:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {rouge1_progress}%"></div>
                        </div>
                    </div>
                    <div class="metric-card {rouge2_class}">
                        <div class="metric-name">ROUGE-2</div>
                        <div class="metric-value">{avg_rouge2:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {rouge2_progress}%"></div>
                        </div>
                    </div>
                    <div class="metric-card {rougeL_class}">
                        <div class="metric-name">ROUGE-L</div>
                        <div class="metric-value">{avg_rougeL:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {rougeL_progress}%"></div>
                        </div>
                    </div>
                </div>

                <h3>BERTScore</h3>
                <div class="metrics-grid">
                    <div class="metric-card {bs_p_class}">
                        <div class="metric-name">Precision</div>
                        <div class="metric-value">{avg_bertscore_precision:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {bs_p_progress}%"></div>
                        </div>
                    </div>
                    <div class="metric-card {bs_r_class}">
                        <div class="metric-name">Recall</div>
                        <div class="metric-value">{avg_bertscore_recall:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {bs_r_progress}%"></div>
                        </div>
                    </div>
                    <div class="metric-card {bs_f1_class}">
                        <div class="metric-name">F1 Score</div>
                        <div class="metric-value">{avg_bertscore_f1:.4f}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {bs_f1_progress}%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detailed Results -->
            <div class="section">
                <h2>Detailed Evaluation Results</h2>
                {paper_details}
            </div>
        </div>

        <div class="footer">
            <p>Generated by Multimodal Summarization Reward Model Evaluation System</p>
            <p>Week 8 Assignment - Multimodal Summarization and Reward Modeling</p>
        </div>
    </div>
</body>
</html>"""

    def _get_metric_class(self, value: float, thresholds: tuple) -> str:
        """Return CSS class based on score value"""
        if value >= thresholds[0]:
            return "excellent"
        elif value >= thresholds[1]:
            return "good"
        else:
            return "average"

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))

    def generate(self, evaluation_results: dict, output_path: str = "evaluation_report_en.html"):
        """Generate HTML report"""
        summary = evaluation_results["summary"]

        # Calculate progress bars and style classes
        reward_progress = min(summary["avg_reward_score"] * 100, 100)

        rouge1 = summary["avg_rouge1"]
        rouge1_progress = min(rouge1 * 100, 100)
        rouge1_class = self._get_metric_class(rouge1, (0.4, 0.3))

        rouge2 = summary["avg_rouge2"]
        rouge2_progress = min(rouge2 * 100, 100)
        rouge2_class = self._get_metric_class(rouge2, (0.2, 0.1))

        rougeL = summary["avg_rougeL"]
        rougeL_progress = min(rougeL * 100, 100)
        rougeL_class = self._get_metric_class(rougeL, (0.35, 0.25))

        bs_p = summary["avg_bertscore_precision"]
        bs_p_progress = min(bs_p * 100, 100)
        bs_p_class = self._get_metric_class(bs_p, (0.75, 0.65))

        bs_r = summary["avg_bertscore_recall"]
        bs_r_progress = min(bs_r * 100, 100)
        bs_r_class = self._get_metric_class(bs_r, (0.75, 0.65))

        bs_f1 = summary["avg_bertscore_f1"]
        bs_f1_progress = min(bs_f1 * 100, 100)
        bs_f1_class = self._get_metric_class(bs_f1, (0.75, 0.65))

        # Generate paper details
        paper_details_html = ""
        for result in evaluation_results["paper_results"]:
            paper_details_html += f"""
            <div class="paper-card">
                <h4>{self._escape_html(result['paper_id'])}</h4>

                <div class="summary-box generated">
                    <div class="summary-label">Generated Summary</div>
                    <div>{self._escape_html(result['generated_summary'])}</div>
                </div>

                {f'''<div class="summary-box reference">
                    <div class="summary-label">Reference Summary</div>
                    <div>{self._escape_html(result['reference_summary'])}</div>
                </div>''' if result.get('reference_summary') else ''}

                <div class="metrics-row">
                    <div class="metric-badge">Reward Score: {result['reward_score']:.4f}</div>
                    {f'''<div class="metric-badge">ROUGE-1: {result['rouge_scores'].get('rouge1', 0):.4f}</div>
                       <div class="metric-badge">ROUGE-2: {result['rouge_scores'].get('rouge2', 0):.4f}</div>
                       <div class="metric-badge">ROUGE-L: {result['rouge_scores'].get('rougeL', 0):.4f}</div>''' if result.get('rouge_scores') else ''}
                    {f'''<div class="metric-badge">BERTScore F1: {result['bertscore'].get('f1', 0):.4f}</div>''' if result.get('bertscore') else ''}
                </div>
            </div>
            """

        # Fill template
        html_content = self.template.format(
            timestamp=evaluation_results["timestamp"],
            total_papers=summary["total_papers"],
            avg_reward_score=summary["avg_reward_score"],
            std_reward_score=summary["std_reward_score"],
            reward_progress=f"{reward_progress:.1f}",
            avg_rouge1=rouge1,
            rouge1_progress=f"{rouge1_progress:.1f}",
            rouge1_class=rouge1_class,
            avg_rouge2=rouge2,
            rouge2_progress=f"{rouge2_progress:.1f}",
            rouge2_class=rouge2_class,
            avg_rougeL=rougeL,
            rougeL_progress=f"{rougeL_progress:.1f}",
            rougeL_class=rougeL_class,
            avg_bertscore_precision=bs_p,
            bs_p_progress=f"{bs_p_progress:.1f}",
            bs_p_class=bs_p_class,
            avg_bertscore_recall=bs_r,
            bs_r_progress=f"{bs_r_progress:.1f}",
            bs_r_class=bs_r_class,
            avg_bertscore_f1=bs_f1,
            bs_f1_progress=f"{bs_f1_progress:.1f}",
            bs_f1_class=bs_f1_class,
            paper_details=paper_details_html
        )

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report saved to: {output_path}")
        return output_path


def main():
    """Generate English report from existing results"""
    json_path = "evaluation_results.json"
    output_path = "evaluation_report_en.html"

    # Load evaluation results
    print(f"Loading results from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Generate English report
    print("Generating English HTML report...")
    generator = EnglishReportGenerator()
    generator.generate(results, output_path)

    print(f"\nReport generated successfully!")
    print(f"Open: {output_path}")


if __name__ == "__main__":
    main()
