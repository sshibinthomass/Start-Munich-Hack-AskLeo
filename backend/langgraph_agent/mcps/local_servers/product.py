import json
from mcp.server.fastmcp import FastMCP
from typing import Optional, List, Union
from pathlib import Path

# Initialize FastMCP server
mcp = FastMCP("product")


class ProductSession:
    def __init__(self):
        self.product_data: Optional[List[dict]] = None
        self.data_file = (
            Path(__file__).parent.parent.parent.parent / "data" / "product.json"
        )
        self._load_product_data()

    def _load_product_data(self):
        """Load product data from JSON file."""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
                # Handle both array and single object formats
                if isinstance(data, list):
                    self.product_data = data
                elif isinstance(data, dict):
                    self.product_data = [data]
                else:
                    self.product_data = None
        except Exception:
            self.product_data = None

    def _find_product_by_id(self, product_id: str) -> Optional[dict]:
        """Find a product by its ID."""
        if not self.product_data:
            return None
        for product in self.product_data:
            if product.get("id") == product_id:
                return product
        return None

    def _get_nested_value(
        self, data: Union[dict, list], path: str
    ) -> Union[dict, list, str, int, float, bool, None]:
        """Get a nested value from data using dot notation.

        Supports paths like:
        - "{productId}.name" - Get name of specific product
        - "{productId}.versions.{versionName}.price" - Get price of specific version
        - "products" - Get all products (if path starts with "products")
        """
        if not data:
            return None

        parts = path.split(".")
        current = data

        # Handle "products" prefix to get all products
        if parts[0] == "products" and isinstance(current, list):
            if len(parts) == 1:
                return current
            # Remove "products" prefix and continue
            parts = parts[1:]
            if not parts:
                return current

        for i, part in enumerate(parts):
            if current is None:
                return None

            if isinstance(current, dict):
                # If we're looking for "versions", get the list
                if part == "versions":
                    current = current.get("versions", [])
                else:
                    current = current.get(part)
            elif isinstance(current, list):
                # If we're in a list, try to find by ID first, then by version name
                found = False
                for item in current:
                    if isinstance(item, dict):
                        # Try matching by ID
                        if item.get("id") == part:
                            current = item
                            found = True
                            break
                        # Try matching by version name
                        elif item.get("version", "").lower() == part.lower():
                            current = item
                            found = True
                            break
                if not found:
                    return None
            else:
                # Can't navigate further
                return None

        return current

    def get_values(self, keys: List[str]) -> str:
        """Get values for one or more keys/paths.

        Args:
            keys: List of key paths to retrieve. Supports:
                - "{productId}.{field}" - Get field from specific product (e.g., "va_prod_001.name")
                - "{productId}.versions.{versionName}.{field}" - Get version field (e.g., "va_prod_001.versions.Gravimetric 3gr.price")
                - "products" - Get all products
                - "products.{productId}.{field}" - Alternative syntax
        """
        if not self.product_data:
            return json.dumps({"error": "No product data available."}, indent=2)

        result = {}

        for key in keys:
            value = self._get_nested_value(self.product_data, key)

            if value is None:
                result[key] = "Not found"
            elif isinstance(value, (dict, list)):
                result[key] = value
            else:
                result[key] = value

        return json.dumps(result, indent=2, default=str)

    def get_product_by_id(self, product_id: str) -> str:
        """Get a complete product by its ID."""
        product = self._find_product_by_id(product_id)
        if not product:
            return json.dumps(
                {"error": f"Product with ID '{product_id}' not found."}, indent=2
            )
        return json.dumps(product, indent=2, default=str)

    def list_products(self) -> str:
        """List all available product IDs and names."""
        if not self.product_data:
            return json.dumps({"error": "No product data available."}, indent=2)

        products = []
        for product in self.product_data:
            products.append(
                {
                    "id": product.get("id"),
                    "name": product.get("name"),
                    "category": product.get("category"),
                    "brand": product.get("brand"),
                }
            )

        return json.dumps(products, indent=2)

    def get_available_keys(self) -> str:
        """Get a list of available keys/paths that can be queried."""
        if not self.product_data:
            return json.dumps({"error": "No product data available."}, indent=2)

        product_keys = []
        version_keys = []

        for product in self.product_data:
            product_id = product.get("id", "")
            if not product_id:
                continue

            # Top-level product fields
            product_keys.append(f"{product_id}.id")
            product_keys.append(f"{product_id}.name")
            product_keys.append(f"{product_id}.category")
            product_keys.append(f"{product_id}.brand")
            if "vendor" in product:
                product_keys.append(f"{product_id}.vendor")
            if "currency" in product:
                product_keys.append(f"{product_id}.currency")
            if "color" in product:
                product_keys.append(f"{product_id}.color")

            # Version fields
            versions = product.get("versions", [])
            for version in versions:
                version_name = version.get("version", "")
                if version_name:
                    base_path = f"{product_id}.versions.{version_name}"
                    version_keys.extend(
                        [
                            f"{base_path}.version",
                            f"{base_path}.productCode",
                            f"{base_path}.description",
                            f"{base_path}.price",
                            f"{base_path}.minPrice",
                            f"{base_path}.maxPrice",
                            f"{base_path}.deliveryTimeDays",
                            f"{base_path}.warranty",
                            f"{base_path}.inStock",
                            f"{base_path}.stockQuantity",
                            f"{base_path}.rating",
                            f"{base_path}.reviews",
                            f"{base_path}.features",
                            f"{base_path}.specifications",
                            f"{base_path}.images",
                        ]
                    )

        return json.dumps(
            {
                "product_keys": product_keys,
                "version_keys": version_keys,
                "all_keys": product_keys + version_keys,
                "note": "You can also use 'products' to get all products, or '{productId}' to get a full product object.",
            },
            indent=2,
        )


session = ProductSession()


@mcp.tool()
async def product_get(keys: List[str]) -> str:
    """Get product data by specifying one or more keys/paths.

    You can request single or multiple values in one call. Use dot notation for nested paths.

    Examples:
        - ["va_prod_001.name"] - Get name of product va_prod_001
        - ["va_prod_001.id", "va_prod_001.category", "va_prod_001.brand"] - Get multiple fields from one product
        - ["va_prod_001.versions.Gravimetric 3gr.price"] - Get price for specific version
        - ["va_prod_001.versions.Gravimetric 3gr.price", "va_prod_001.versions.Gravimetric 3gr.features"] - Get multiple version fields
        - ["products"] - Get all products
        - ["va_prod_002.versions.Digit 3gr.minPrice", "va_prod_002.versions.Digit 3gr.maxPrice"] - Get price range

    Path format:
        - "{productId}.{field}" - Get top-level field from product
        - "{productId}.versions.{versionName}.{field}" - Get field from specific version
        - "products" - Get all products array

    Args:
        keys: List of key paths to retrieve. Each key uses dot notation to navigate the product structure.
    """
    return session.get_values(keys)


@mcp.tool()
async def product_get_by_id(product_id: str) -> str:
    """Get a complete product object by its ID.

    This returns the full product including all versions and their details.

    Args:
        product_id: The product ID (e.g., "va_prod_001", "va_prod_002", "va_prod_003").
    """
    return session.get_product_by_id(product_id)


@mcp.tool()
async def product_list() -> str:
    """List all available products with their IDs, names, categories, and brands.

    This helps discover what products are available before querying specific details.
    """
    return session.list_products()


@mcp.tool()
async def product_list_keys() -> str:
    """List all available keys/paths that can be queried using product_get.

    This helps discover what data is available and the exact path format to use.
    """
    return session.get_available_keys()


if __name__ == "__main__":
    mcp.run(transport="stdio")
