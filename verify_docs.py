#!/usr/bin/env python3
"""
Verify documentation integrity and check for broken links.
"""
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from urllib.parse import urlparse
import requests
import yaml
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocVerifier:
    """Verify documentation integrity."""
    
    def __init__(self):
        """Initialize verifier."""
        self.root_dir = Path(__file__).parent
        self.docs = {}
        self.links = set()
        self.errors = []
        self.warnings = []

    def verify_all(self) -> bool:
        """Run all verification checks."""
        success = all([
            self._check_required_files(),
            self._verify_markdown_links(),
            self._check_code_references(),
            self._verify_yaml_files(),
            self._check_consistency()
        ])
        
        self._print_report()
        return success

    def _check_required_files(self) -> bool:
        """Check if all required documentation files exist."""
        required_files = [
            'README.md',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'CHANGELOG.md',
            'CONTRIBUTORS.md',
            'LICENSE',
            '.github/ISSUE_TEMPLATE/bug_report.md',
            '.github/ISSUE_TEMPLATE/feature_request.md',
            '.github/pull_request_template.md'
        ]
        
        missing = []
        for file in required_files:
            if not (self.root_dir / file).exists():
                missing.append(file)
                
        if missing:
            self.errors.append(f"Missing required files: {', '.join(missing)}")
            return False
        return True

    def _verify_markdown_links(self) -> bool:
        """Verify all markdown links are valid."""
        valid = True
        for md_file in self.root_dir.glob('**/*.md'):
            content = md_file.read_text()
            
            # Internal links
            internal_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            for text, link in internal_links:
                if not link.startswith(('http://', 'https://', 'mailto:')):
                    link_path = self.root_dir / link
                    if not link_path.exists():
                        self.errors.append(f"Broken internal link in {md_file}: {link}")
                        valid = False
            
            # External links (optional check)
            external_links = [
                link for _, link in internal_links
                if link.startswith(('http://', 'https://'))
            ]
            self._verify_external_links(external_links)
            
        return valid

    def _verify_external_links(self, links: List[str]) -> None:
        """Verify external links are accessible."""
        for link in links:
            try:
                response = requests.head(link, timeout=5)
                if response.status_code >= 400:
                    self.warnings.append(f"External link may be broken: {link}")
            except requests.RequestException:
                self.warnings.append(f"Could not verify external link: {link}")

    def _check_code_references(self) -> bool:
        """Check if code references in documentation exist."""
        valid = True
        for md_file in self.root_dir.glob('**/*.md'):
            content = md_file.read_text()
            
            # Find Python file references
            python_files = re.findall(r'`([^`]+\.py)`', content)
            for py_file in python_files:
                if not any(self.root_dir.glob(f'**/{py_file}')):
                    self.warnings.append(f"Referenced Python file not found: {py_file}")
                    valid = False
                    
        return valid

    def _verify_yaml_files(self) -> bool:
        """Verify YAML files are valid."""
        valid = True
        for yaml_file in self.root_dir.glob('**/*.yaml'):
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {yaml_file}: {str(e)}")
                valid = False
        return valid

    def _check_consistency(self) -> bool:
        """Check documentation consistency."""
        valid = True
        
        # Check version consistency
        versions = set()
        version_files = ['setup.py', 'CHANGELOG.md']
        for file in version_files:
            file_path = self.root_dir / file
            if file_path.exists():
                content = file_path.read_text()
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)', content)
                if version_match:
                    versions.add(version_match.group(1))
        
        if len(versions) > 1:
            self.errors.append(f"Inconsistent versions found: {versions}")
            valid = False
        
        # Check documentation references
        references = {
            'CONTRIBUTING.md': ['CODE_OF_CONDUCT.md'],
            'README.md': ['CONTRIBUTING.md', 'LICENSE']
        }
        
        for source, required_refs in references.items():
            source_path = self.root_dir / source
            if source_path.exists():
                content = source_path.read_text()
                for ref in required_refs:
                    if ref not in content:
                        self.warnings.append(
                            f"{source} should reference {ref}"
                        )
        
        return valid

    def _print_report(self) -> None:
        """Print verification report."""
        print("\nDocumentation Verification Report")
        print("================================")
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"❌ {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"⚠️  {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All documentation checks passed!")
        
        print("\nSummary:")
        print(f"- Errors: {len(self.errors)}")
        print(f"- Warnings: {len(self.warnings)}")

def main():
    """Main execution function."""
    verifier = DocVerifier()
    success = verifier.verify_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()