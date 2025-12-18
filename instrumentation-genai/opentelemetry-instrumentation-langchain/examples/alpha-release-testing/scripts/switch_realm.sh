#!/bin/bash
# Alpha Release Testing - Realm Switching Script
# Easily switch between lab0, rc0, and us1 configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_DIR/config"

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
${BLUE}Alpha Release Testing - Realm Switcher${NC}

Usage: $0 [REALM]

Available Realms:
  ${GREEN}lab0${NC}  - Development/Testing environment (default for Alpha)
  ${GREEN}rc0${NC}   - Release Candidate environment
  ${GREEN}us1${NC}   - Production environment

Examples:
  $0 lab0    # Switch to lab0 realm
  $0 rc0     # Switch to rc0 realm
  $0 us1     # Switch to us1 realm

Current Configuration:
  $(if [ -f "$CONFIG_DIR/.env" ]; then
      CURRENT_REALM=$(grep "^SPLUNK_REALM=" "$CONFIG_DIR/.env" 2>/dev/null | cut -d'=' -f2)
      if [ -n "$CURRENT_REALM" ]; then
          echo "Active Realm: ${GREEN}$CURRENT_REALM${NC}"
      else
          echo "Active Realm: ${YELLOW}Unknown${NC}"
      fi
  else
      echo "No active configuration"
  fi)

EOF
}

# Function to validate realm
validate_realm() {
    local realm=$1
    case $realm in
        lab0|rc0|us1)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to backup current config
backup_config() {
    if [ -f "$CONFIG_DIR/.env" ]; then
        local backup_file="$CONFIG_DIR/.env.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$CONFIG_DIR/.env" "$backup_file"
        print_info "Backed up current config to: $(basename $backup_file)"
    fi
}

# Function to switch realm
switch_realm() {
    local realm=$1
    local template_file="$CONFIG_DIR/.env.$realm.template"
    local target_file="$CONFIG_DIR/.env"
    
    # Check if template exists
    if [ ! -f "$template_file" ]; then
        print_error "Template file not found: $template_file"
        exit 1
    fi
    
    # Backup current config
    backup_config
    
    # Copy template to .env
    cp "$template_file" "$target_file"
    print_success "Switched to $realm realm"
    
    # Display realm information
    echo ""
    print_info "Realm Configuration:"
    echo "  Realm: $realm"
    
    # Extract and display key information
    local splunk_realm=$(grep "^SPLUNK_REALM=" "$target_file" | cut -d'=' -f2)
    local splunk_url=$(grep "^SPLUNK_HEC_URL=" "$target_file" | cut -d'=' -f2)
    local otel_endpoint=$(grep "^OTEL_EXPORTER_OTLP_ENDPOINT=" "$target_file" | cut -d'=' -f2)
    local service_name=$(grep "^OTEL_SERVICE_NAME=" "$target_file" | cut -d'=' -f2)
    
    echo "  Splunk Realm: $splunk_realm"
    echo "  HEC URL: $splunk_url"
    echo "  OTEL Endpoint: $otel_endpoint"
    echo "  Service Name: $service_name"
    
    # Check for credentials that need to be updated
    echo ""
    if [ "$realm" != "lab0" ]; then
        print_warning "Action Required: Update credentials in $target_file"
        echo "  Required variables:"
        echo "    - SPLUNK_ACCESS_TOKEN"
        echo "    - SPLUNK_HEC_TOKEN"
        echo "    - AZURE_OPENAI_API_KEY (if using Azure OpenAI)"
    else
        print_success "lab0 credentials are pre-configured"
        print_warning "Update AZURE_OPENAI_API_KEY if testing Azure OpenAI"
    fi
    
    # Offer to open config file
    echo ""
    read -p "Open config file for editing? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-vim} "$target_file"
    fi
}

# Function to show current configuration
show_current_config() {
    local config_file="$CONFIG_DIR/.env"
    
    if [ ! -f "$config_file" ]; then
        print_warning "No active configuration found"
        echo "Run: $0 [lab0|rc0|us1] to set up a realm"
        return
    fi
    
    echo ""
    print_info "Current Configuration:"
    echo ""
    
    # Extract key variables
    local realm=$(grep "^SPLUNK_REALM=" "$config_file" | cut -d'=' -f2)
    local service=$(grep "^OTEL_SERVICE_NAME=" "$config_file" | cut -d'=' -f2)
    local endpoint=$(grep "^OTEL_EXPORTER_OTLP_ENDPOINT=" "$config_file" | cut -d'=' -f2)
    
    echo "  Realm: ${GREEN}$realm${NC}"
    echo "  Service: $service"
    echo "  OTEL Endpoint: $endpoint"
    
    # Check if credentials are configured
    echo ""
    print_info "Credential Status:"
    
    local access_token=$(grep "^SPLUNK_ACCESS_TOKEN=" "$config_file" | cut -d'=' -f2)
    local hec_token=$(grep "^SPLUNK_HEC_TOKEN=" "$config_file" | cut -d'=' -f2)
    local azure_key=$(grep "^AZURE_OPENAI_API_KEY=" "$config_file" | cut -d'=' -f2)
    
    if [[ "$access_token" == *"your-"* ]] || [ -z "$access_token" ]; then
        echo "  SPLUNK_ACCESS_TOKEN: ${RED}Not configured${NC}"
    else
        echo "  SPLUNK_ACCESS_TOKEN: ${GREEN}Configured${NC}"
    fi
    
    if [[ "$hec_token" == *"your-"* ]] || [ -z "$hec_token" ]; then
        echo "  SPLUNK_HEC_TOKEN: ${RED}Not configured${NC}"
    else
        echo "  SPLUNK_HEC_TOKEN: ${GREEN}Configured${NC}"
    fi
    
    if [[ "$azure_key" == *"your-"* ]] || [ -z "$azure_key" ]; then
        echo "  AZURE_OPENAI_API_KEY: ${YELLOW}Not configured${NC}"
    else
        echo "  AZURE_OPENAI_API_KEY: ${GREEN}Configured${NC}"
    fi
    
    echo ""
}

# Main script logic
main() {
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        usage
        show_current_config
        exit 0
    fi
    
    local realm=$1
    
    # Validate realm
    if ! validate_realm "$realm"; then
        print_error "Invalid realm: $realm"
        echo ""
        usage
        exit 1
    fi
    
    # Switch to realm
    switch_realm "$realm"
    
    echo ""
    print_success "Realm switch complete!"
    echo ""
    print_info "Next steps:"
    echo "  1. Verify credentials in: $CONFIG_DIR/.env"
    echo "  2. Load environment: source $CONFIG_DIR/.env"
    echo "  3. Run tests: pytest tests/ -v"
    echo ""
}

# Run main function
main "$@"
