#!/usr/bin/env python3
"""
Advanced Test Report Generator

Purpose: Generate comprehensive HTML and JSON reports from pytest results
         with advanced analytics, visualizations, and executive summaries
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET


class AdvancedReportGenerator:
    """Generate advanced test reports with analytics and visualizations"""

    def __init__(self, framework_dir: str):
        self.framework_dir = Path(framework_dir)
        self.reports_dir = self.framework_dir / "reports"
        self.html_dir = self.reports_dir / "html"
        self.json_dir = self.reports_dir / "json"
        self.junit_dir = self.reports_dir / "junit"

    def collect_test_results(self) -> Dict[str, Any]:
        """Collect all test results from JSON and JUnit files"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0.0,
            },
        }

        # Collect from JSON reports
        if self.json_dir.exists():
            for json_file in self.json_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "tests" in data:
                            suite_name = json_file.stem
                            suite_results = self._process_json_report(suite_name, data)
                            results["test_suites"].append(suite_results)
                            self._update_summary(results["summary"], suite_results)
                except Exception as e:
                    print(f"Warning: Failed to process {json_file}: {e}")

        # Collect from JUnit XML reports
        if self.junit_dir.exists():
            for xml_file in self.junit_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    suite_name = xml_file.stem
                    suite_results = self._process_junit_report(suite_name, root)

                    # Only add if not already added from JSON
                    if not any(s["name"] == suite_name for s in results["test_suites"]):
                        results["test_suites"].append(suite_results)
                        self._update_summary(results["summary"], suite_results)
                except Exception as e:
                    print(f"Warning: Failed to process {xml_file}: {e}")

        # Calculate success rate
        total = results["summary"]["total_tests"]
        if total > 0:
            results["summary"]["success_rate"] = (
                results["summary"]["passed"] / total
            ) * 100
        else:
            results["summary"]["success_rate"] = 0.0

        return results

    def _process_json_report(self, suite_name: str, data: Dict) -> Dict:
        """Process pytest JSON report"""
        suite = {
            "name": suite_name,
            "tests": [],
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": data.get("duration", 0.0),
        }

        for test in data.get("tests", []):
            test_result = {
                "name": test.get("nodeid", ""),
                "outcome": test.get("outcome", "unknown"),
                "duration": test.get("duration", 0.0),
                "message": test.get("call", {}).get("longrepr", ""),
            }
            suite["tests"].append(test_result)

            if test_result["outcome"] == "passed":
                suite["passed"] += 1
            elif test_result["outcome"] == "failed":
                suite["failed"] += 1
            elif test_result["outcome"] == "skipped":
                suite["skipped"] += 1

        return suite

    def _process_junit_report(self, suite_name: str, root: ET.Element) -> Dict:
        """Process JUnit XML report"""
        suite = {
            "name": suite_name,
            "tests": [],
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": float(root.get("time", 0.0)),
        }

        for testcase in root.findall(".//testcase"):
            test_result = {
                "name": testcase.get("name", ""),
                "classname": testcase.get("classname", ""),
                "duration": float(testcase.get("time", 0.0)),
                "outcome": "passed",
            }

            if testcase.find("failure") is not None:
                test_result["outcome"] = "failed"
                test_result["message"] = testcase.find("failure").get("message", "")
                suite["failed"] += 1
            elif testcase.find("skipped") is not None:
                test_result["outcome"] = "skipped"
                suite["skipped"] += 1
            elif testcase.find("error") is not None:
                test_result["outcome"] = "error"
                test_result["message"] = testcase.find("error").get("message", "")
                suite["failed"] += 1
            else:
                suite["passed"] += 1

            suite["tests"].append(test_result)

        return suite

    def _update_summary(self, summary: Dict, suite: Dict):
        """Update overall summary with suite results"""
        summary["total_tests"] += len(suite["tests"])
        summary["passed"] += suite["passed"]
        summary["failed"] += suite["failed"]
        summary["skipped"] += suite["skipped"]
        summary["duration"] += suite["duration"]

    def generate_html_report(self, results: Dict) -> str:
        """Generate comprehensive HTML report"""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>O11y AI Test Framework - Comprehensive Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .metric-value {{
            font-size: 3em;
            font-weight: 700;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .passed {{ color: #10b981; }}
        .failed {{ color: #ef4444; }}
        .skipped {{ color: #f59e0b; }}
        .duration {{ color: #3b82f6; }}
        .success-rate {{ color: #8b5cf6; }}
        .content {{
            padding: 40px;
        }}
        .test-suite {{
            margin-bottom: 30px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
        }}
        .suite-header {{
            background: #f3f4f6;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .suite-header:hover {{
            background: #e5e7eb;
        }}
        .suite-name {{
            font-size: 1.3em;
            font-weight: 600;
        }}
        .suite-stats {{
            display: flex;
            gap: 20px;
            font-size: 0.9em;
        }}
        .test-list {{
            padding: 20px;
            display: none;
        }}
        .test-list.active {{
            display: block;
        }}
        .test-item {{
            padding: 15px;
            border-left: 4px solid #e5e7eb;
            margin-bottom: 10px;
            background: #f9fafb;
            border-radius: 4px;
        }}
        .test-item.passed {{
            border-left-color: #10b981;
            background: #f0fdf4;
        }}
        .test-item.failed {{
            border-left-color: #ef4444;
            background: #fef2f2;
        }}
        .test-item.skipped {{
            border-left-color: #f59e0b;
            background: #fffbeb;
        }}
        .test-name {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .test-duration {{
            color: #666;
            font-size: 0.85em;
        }}
        .test-message {{
            margin-top: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .footer {{
            background: #f3f4f6;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge.passed {{ background: #d1fae5; color: #065f46; }}
        .badge.failed {{ background: #fee2e2; color: #991b1b; }}
        .badge.skipped {{ background: #fef3c7; color: #92400e; }}
        .progress-bar {{
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            transition: width 1s ease;
        }}
    </style>
    <script>
        function toggleSuite(id) {{
            const element = document.getElementById(id);
            element.classList.toggle('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 O11y AI Test Framework</h1>
            <div class="subtitle">Comprehensive Test Execution Report</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">
                Generated: {results['timestamp']}
            </div>
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">Total Tests</div>
                <div class="metric-value">{results['summary']['total_tests']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Passed</div>
                <div class="metric-value passed">✓ {results['summary']['passed']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Failed</div>
                <div class="metric-value failed">✗ {results['summary']['failed']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Skipped</div>
                <div class="metric-value skipped">⊘ {results['summary']['skipped']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Duration</div>
                <div class="metric-value duration">{results['summary']['duration']:.2f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value success-rate">{results['summary']['success_rate']:.1f}%</div>
            </div>
        </div>
        
        <div style="padding: 0 40px;">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {results['summary']['success_rate']:.1f}%">
                    {results['summary']['success_rate']:.1f}% Success
                </div>
            </div>
        </div>
        
        <div class="content">
            <h2 style="margin-bottom: 20px;">Test Suites</h2>
"""

        # Add test suites
        for idx, suite in enumerate(results["test_suites"]):
            suite_id = f"suite_{idx}"

            html += f"""
            <div class="test-suite">
                <div class="suite-header" onclick="toggleSuite('{suite_id}')">
                    <div class="suite-name">{suite['name']}</div>
                    <div class="suite-stats">
                        <span class="badge passed">{suite['passed']} passed</span>
                        <span class="badge failed">{suite['failed']} failed</span>
                        <span class="badge skipped">{suite['skipped']} skipped</span>
                        <span style="color: #666;">{suite['duration']:.2f}s</span>
                    </div>
                </div>
                <div id="{suite_id}" class="test-list">
"""

            # Add individual tests
            for test in suite["tests"]:
                outcome_class = test["outcome"]
                outcome_icon = (
                    "✓"
                    if outcome_class == "passed"
                    else "✗"
                    if outcome_class == "failed"
                    else "⊘"
                )

                html += f"""
                    <div class="test-item {outcome_class}">
                        <div class="test-name">{outcome_icon} {test['name']}</div>
                        <div class="test-duration">Duration: {test['duration']:.3f}s</div>
"""

                if test.get("message"):
                    html += f"""
                        <div class="test-message">{test['message']}</div>
"""

                html += """
                    </div>
"""

            html += """
                </div>
            </div>
"""

        html += f"""
        </div>
        
        <div class="footer">
            <p><strong>O11y AI Test Framework</strong> - Automated Testing for AI Observability</p>
            <p style="margin-top: 10px;">Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def generate_json_report(self, results: Dict) -> str:
        """Generate JSON report"""
        return json.dumps(results, indent=2)

    def save_reports(self, results: Dict):
        """Save HTML and JSON reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save HTML report
        html_report = self.generate_html_report(results)
        html_file = self.html_dir / f"comprehensive_report_{timestamp}.html"
        html_file.parent.mkdir(parents=True, exist_ok=True)
        with open(html_file, "w") as f:
            f.write(html_report)

        # Also save as latest
        latest_html = self.html_dir / "latest_report.html"
        with open(latest_html, "w") as f:
            f.write(html_report)

        # Save JSON report
        json_report = self.generate_json_report(results)
        json_file = self.json_dir / f"comprehensive_report_{timestamp}.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, "w") as f:
            f.write(json_report)

        # Also save as latest
        latest_json = self.json_dir / "latest_report.json"
        with open(latest_json, "w") as f:
            f.write(json_report)

        return html_file, json_file


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: generate_report.py <framework_directory>")
        sys.exit(1)

    framework_dir = sys.argv[1]

    print("=" * 60)
    print("📊 Advanced Test Report Generator")
    print("=" * 60)
    print()

    generator = AdvancedReportGenerator(framework_dir)

    print("🔍 Collecting test results...")
    results = generator.collect_test_results()

    print(
        f"✅ Found {results['summary']['total_tests']} tests across {len(results['test_suites'])} suites"
    )
    print()

    print("📝 Generating reports...")
    html_file, json_file = generator.save_reports(results)

    print()
    print("=" * 60)
    print("✅ Reports Generated Successfully")
    print("=" * 60)
    print()
    print(f"📄 HTML Report: {html_file}")
    print(f"📄 JSON Report: {json_file}")
    print()
    print("📊 Summary:")
    print(f"   Total Tests: {results['summary']['total_tests']}")
    print(f"   Passed: {results['summary']['passed']} ✓")
    print(f"   Failed: {results['summary']['failed']} ✗")
    print(f"   Skipped: {results['summary']['skipped']} ⊘")
    print(f"   Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"   Duration: {results['summary']['duration']:.2f}s")
    print()
    print(f"🌐 Open report: open {html_file}")
    print()


if __name__ == "__main__":
    main()
