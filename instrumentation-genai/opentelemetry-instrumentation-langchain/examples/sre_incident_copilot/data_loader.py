"""Data loading utilities for SRE Incident Copilot."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class DataLoader:
    """Loads seeded data for the incident copilot."""

    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self._service_catalog: Optional[Dict] = None
        self._alert_catalog: Optional[Dict] = None

    def load_service_catalog(self) -> Dict:
        """Load service catalog from JSON file."""
        if self._service_catalog is None:
            catalog_path = self.data_dir / "service_catalog.json"
            with open(catalog_path, "r") as f:
                self._service_catalog = json.load(f)
        return self._service_catalog

    def get_service(self, service_id: str) -> Optional[Dict]:
        """Get service by ID."""
        catalog = self.load_service_catalog()
        for service in catalog.get("services", []):
            if service["id"] == service_id:
                return service
        return None

    def load_alert_catalog(self) -> Dict:
        """Load alert catalog from JSON file."""
        if self._alert_catalog is None:
            catalog_path = self.data_dir / "alert_catalog.json"
            with open(catalog_path, "r") as f:
                self._alert_catalog = json.load(f)
        return self._alert_catalog

    def get_alert(self, alert_id: str) -> Optional[Dict]:
        """Get alert by ID."""
        catalog = self.load_alert_catalog()
        for alert in catalog.get("alerts", []):
            if alert["id"] == alert_id:
                return alert
        return None

    def get_alert_by_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Get alert by scenario ID."""
        catalog = self.load_alert_catalog()
        for alert in catalog.get("alerts", []):
            if alert.get("scenario_id") == scenario_id:
                return alert
        return None

    def load_runbook(self, runbook_name: str) -> Optional[str]:
        """Load runbook markdown file.

        Args:
            runbook_name: Name of runbook file (without .md extension)

        Returns:
            Runbook content as string, or None if not found
        """
        runbook_path = self.data_dir / "runbooks" / f"{runbook_name}.md"
        if runbook_path.exists():
            with open(runbook_path, "r") as f:
                return f.read()
        return None

    def list_runbooks(self) -> List[str]:
        """List all available runbook names."""
        runbooks_dir = self.data_dir / "runbooks"
        if not runbooks_dir.exists():
            return []
        return [f.stem for f in runbooks_dir.glob("*.md")]

    def get_all_alerts(self) -> List[Dict]:
        """Get all alerts from catalog."""
        catalog = self.load_alert_catalog()
        return catalog.get("alerts", [])
