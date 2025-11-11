import os
import copy
import requests
import urllib.parse
from .base_tool import BaseTool
from .tool_registry import register_tool

# ---- Helper: human readable -> openFDA code mapping ----
HUMAN_TO_FDA_MAP = {
    "fulfillexpeditecriteria": {"Yes": "1", "No": "2"},
    "patient.patientsex": {"Unknown": "0", "Male": "1", "Female": "2"},
    "patient.patientagegroup": {
        "Neonate": "1",
        "Infant": "2",
        "Child": "3",
        "Adolescent": "4",
        "Adult": "5",
        "Elderly": "6",
    },
    "patientonsetageunit": {
        "Decade": "800",
        "Year": "801",
        "Month": "802",
        "Week": "803",
        "Day": "804",
        "Hour": "805",
    },
    "patient.reaction.reactionoutcome": {
        "Recovered/resolved": "1",
        "Recovering/resolving": "2",
        "Not recovered/not resolved": "3",
        "Recovered/resolved with sequelae": "4",
        "Fatal": "5",
        "Unknown": "6",
    },
    "serious": {"Yes": "1", "No": "2"},
    "seriousnessdeath": {"Yes": "1"},
    "seriousnesshospitalization": {"Yes": "1"},
    "seriousnessdisabling": {"Yes": "1"},
    "seriousnesslifethreatening": {"Yes": "1"},
    "seriousnessother": {"Yes": "1"},
    "primarysource.qualification": {
        "Physician": "1",
        "Pharmacist": "2",
        "Other health professional": "3",
        "Lawyer": "4",
        "Consumer or non-health professional": "5",
    },
    "patient.drug.drugcharacterization": {
        "Suspect": "1",
        "Concomitant": "2",
        "Interacting": "3",
    },
    "patient.drug.drugadministrationroute": {
        "Oral": "048",
        "Intravenous": "042",
        "Intramuscular": "030",
        "Subcutaneous": "058",
        "Rectal": "054",
        "Topical": "061",
        "Respiratory (inhalation)": "055",
        "Ophthalmic": "047",
        "Unknown": "065",
    },
}


# ---- Base Tool Class ----
@register_tool("FDADrugAdverseEventTool")
class FDADrugAdverseEventTool(BaseTool):
    def __init__(
        self,
        tool_config,
        endpoint_url="https://api.fda.gov/drug/event.json",
        api_key=None,
    ):
        super().__init__(tool_config)
        self.endpoint_url = endpoint_url
        self.api_key = api_key or os.getenv("FDA_API_KEY")
        self.search_fields = tool_config.get("fields", {}).get("search_fields", {})
        self.return_fields = tool_config.get("fields", {}).get("return_fields", [])
        self.count_field = tool_config.get("count_field") or (
            self.return_fields[0] if self.return_fields else None
        )
        self.return_fields_mapping = tool_config.get("fields", {}).get(
            "return_fields_mapping", {}
        )

        if not self.count_field:
            raise ValueError(
                "Either 'count_field' or 'return_fields' must be defined in tool_config."
            )

        # Store allowed enum values
        self.parameter_enums = {}
        if "parameter" in tool_config and "properties" in tool_config["parameter"]:
            for param_name, param_def in tool_config["parameter"]["properties"].items():
                if "enum" in param_def:
                    self.parameter_enums[param_name] = param_def["enum"]

    def run(self, arguments):
        arguments = copy.deepcopy(arguments)

        # Validate enum parameters
        validation_error = self.validate_enum_arguments(arguments)
        if validation_error:
            return {"error": validation_error}

        response = self._search(arguments)
        return self._post_process(response)

    def validate_enum_arguments(self, arguments):
        """Validate that enum-based arguments match the allowed values"""
        for param_name, value in arguments.items():
            if param_name in self.parameter_enums and value is not None:
                allowed_values = self.parameter_enums[param_name]
                if value not in allowed_values:
                    return f"Invalid value '{value}' for parameter '{param_name}'. Allowed values are: {', '.join(allowed_values)}"
        return None

    def _post_process(self, response):
        if not response or not isinstance(response, list):
            return []

        if not self.return_fields_mapping:
            return response

        mapped_results = []
        for item in response:
            try:
                term = item.get("term")
                count = item.get("count", 0)
                mapped_term = self.return_fields_mapping.get(self.count_field, {}).get(
                    str(term), term
                )
                mapped_results.append({"term": mapped_term, "count": count})
            except Exception:
                # Keep the original term in case of an exception
                mapped_results.append(item)

        return mapped_results

    def _search(self, arguments):
        search_parts = []
        for param_name, value in arguments.items():
            fda_fields = self.search_fields.get(
                param_name, [param_name]
            )  # Map param -> FDA field
            # Use the first field name for value mapping
            fda_field = fda_fields[0] if fda_fields else param_name

            # Apply value mapping using FDA field name
            # (for proper enum mapping)
            mapping_error, mapped_value = self._map_value(fda_field, value)
            if mapping_error:
                return [{"error": mapping_error}]
            if mapped_value is None:
                continue  # Skip this field if instructed

            # Build search parts using FDA field name(s)
            for fda_field_name in fda_fields:
                if isinstance(mapped_value, str) and " " in mapped_value:
                    search_parts.append(f'{fda_field_name}:"{mapped_value}"')
                else:
                    search_parts.append(f"{fda_field_name}:{mapped_value}")

        # Final search query
        search_query = "+AND+".join(search_parts)
        search_encoded = urllib.parse.quote(search_query, safe='+:"')

        # Build URL
        if self.api_key:
            url = f"{self.endpoint_url}?api_key={self.api_key}&search={search_encoded}&count={self.count_field}"
        else:
            url = (
                f"{self.endpoint_url}?search={search_encoded}&count={self.count_field}"
            )

        # API request
        try:
            response = requests.get(url)
            # Handle 404 as "no matches found" - return empty list instead of error
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    if "error" in error_data and "No matches found" in str(
                        error_data.get("error", {})
                    ):
                        return []  # Return empty list for no matches
                except (ValueError, KeyError):
                    pass
            response.raise_for_status()
            response = response.json()
            if "results" in response:
                response = response["results"]
            return response
        except requests.exceptions.RequestException as e:
            return [{"error": f"API request failed: {str(e)}"}]

    def _map_value(self, param_name, value):
        # Special handling for seriousness fields: if value is "No", skip this field
        seriousness_fields = {
            "seriousnessdeath",
            "seriousnesshospitalization",
            "seriousnessdisabling",
            "seriousnesslifethreatening",
            "seriousnessother",
        }
        if param_name in seriousness_fields:
            if value == "No":
                return None, None  # Signal to skip this field
            if value == "Yes":
                return None, "1"
            # If not Yes/No, error
            return (
                f"Invalid value '{value}' for '{param_name}'. Allowed values: ['Yes', 'No']",
                None,
            )

        if param_name in HUMAN_TO_FDA_MAP:
            value_map = HUMAN_TO_FDA_MAP[param_name]
            if value not in value_map:
                print("No mapping found for value:", value, "skipping")
                allowed_values = list(value_map.keys())
                return (
                    f"Invalid value '{value}' for '{param_name}'. Allowed values: {allowed_values}",
                    None,
                )
            return None, value_map[value]
        return None, value


@register_tool("FDACountAdditiveReactionsTool")
class FDACountAdditiveReactionsTool(FDADrugAdverseEventTool):
    """
    Leverage openFDA API to count adverse reaction events across multiple drugs in one request.
    """

    def __init__(
        self,
        tool_config,
        endpoint_url="https://api.fda.gov/drug/event.json",
        api_key=None,
    ):
        super().__init__(tool_config)

    def run(self, arguments):
        # Make a copy to avoid modifying the original
        arguments = copy.deepcopy(arguments)

        # Validate medicinalproducts list first
        drugs = arguments.pop("medicinalproducts", [])
        if not drugs:
            return {"error": "`medicinalproducts` list is required."}
        if not isinstance(drugs, list):
            return {"error": "`medicinalproducts` must be a list of drug names."}

        # Validate the remaining enum parameters
        validation_error = self.validate_enum_arguments(arguments)
        if validation_error:
            return {"error": validation_error}

        # Build OR clause for multiple drugs
        escaped = []
        for d in drugs:
            val = urllib.parse.quote(d, safe="")
            escaped.append(f"patient.drug.medicinalproduct:{val}")
        or_clause = "+OR+".join(escaped)

        # Combine additional filters
        filters = []
        for k, v in arguments.items():
            # Get FDA field name(s) from search_fields mapping
            fda_fields = self.search_fields.get(k, [k])
            # Use the first field name for value mapping
            fda_field = fda_fields[0] if fda_fields else k

            # Map value using FDA field name (for proper enum mapping)
            mapping_error, mapped = self._map_value(fda_field, v)
            if mapping_error:
                return {"error": mapping_error}
            if mapped is None:
                continue  # Skip this field if instructed

            # Use FDA field name(s) in the query
            for fda_field_name in fda_fields:
                if isinstance(mapped, str) and " " in mapped:
                    filters.append(f'{fda_field_name}:"{mapped}"')
                else:
                    filters.append(f"{fda_field_name}:{mapped}")

        filter_str = "+AND+".join(filters) if filters else ""
        search_query = (
            f"({or_clause})" + (f"+AND+{filter_str}" if filter_str else "")
        )
        # URL encode the search query, preserving +, :, and " as safe chars
        search_encoded = urllib.parse.quote(search_query, safe='+:"')

        # Call API
        if self.api_key:
            url = (
                f"{self.endpoint_url}?api_key={self.api_key}"
                f"&search={search_encoded}&count={self.count_field}"
            )
        else:
            url = (
                f"{self.endpoint_url}?search={search_encoded}"
                f"&count={self.count_field}"
            )

        try:
            resp = requests.get(url)
            # Handle 404 as "no matches found" - return empty list instead of error
            if resp.status_code == 404:
                try:
                    error_data = resp.json()
                    if "error" in error_data and "No matches found" in str(
                        error_data.get("error", {})
                    ):
                        return []  # Return empty list for no matches
                except (ValueError, KeyError):
                    pass
            resp.raise_for_status()
            results = resp.json().get("results", [])
            results = self._post_process(results)
            return results
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
