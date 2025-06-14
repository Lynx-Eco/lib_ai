use lib_ai::agent::StructuredProvider;
use lib_ai::JsonSchema;
use serde::{Deserialize, Serialize};

// Example 1: Simple Person struct
#[derive(Debug, Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
    email: String,
}

impl StructuredProvider for Person {
    fn schema() -> JsonSchema {
        JsonSchema {
            name: "Person".to_string(),
            description: Some("A person's information".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Full name" },
                    "age": { "type": "integer", "minimum": 0, "maximum": 150 },
                    "email": { "type": "string", "format": "email" }
                },
                "required": ["name", "age", "email"]
            }),
            strict: Some(true),
        }
    }
}

// Example 2: Manual implementation for Product
#[derive(Debug, Serialize, Deserialize, Default)]
struct Product {
    id: String,
    name: String,
    price: f64,
    in_stock: bool,
}

impl StructuredProvider for Product {
    fn schema() -> JsonSchema {
        JsonSchema {
            name: "Product".to_string(),
            description: Some("Product information".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Product ID" },
                    "name": { "type": "string", "description": "Product name" },
                    "price": { "type": "number", "minimum": 0 },
                    "in_stock": { "type": "boolean" }
                },
                "required": ["id", "name", "price", "in_stock"]
            }),
            strict: Some(true),
        }
    }
}

// Example 3: Nested structures
#[derive(Debug, Serialize, Deserialize)]
struct Order {
    order_id: String,
    customer: Person,
    items: Vec<OrderItem>,
    total: f64,
    status: OrderStatus,
}

#[derive(Debug, Serialize, Deserialize)]
struct OrderItem {
    product_id: String,
    quantity: u32,
    price: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OrderStatus {
    Pending,
    Processing,
    Shipped,
    Delivered,
    Cancelled,
}

impl StructuredProvider for Order {
    fn schema() -> JsonSchema {
        JsonSchema {
            name: "Order".to_string(),
            description: Some("An order with customer and items".to_string()),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Unique order identifier"
                    },
                    "customer": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "age": { "type": "integer" },
                            "email": { "type": "string" }
                        },
                        "required": ["name", "age", "email"]
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": { "type": "string" },
                                "quantity": { "type": "integer", "minimum": 1 },
                                "price": { "type": "number" }
                            },
                            "required": ["product_id", "quantity", "price"]
                        }
                    },
                    "total": {
                        "type": "number",
                        "description": "Total order amount"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]
                    }
                },
                "required": ["order_id", "customer", "items", "total", "status"]
            }),
            strict: Some(true),
        }
    }
}

fn main() {
    println!("Structured Output Schema Examples");
    println!("=================================\n");

    // Example 1: Simple schema
    println!("1. Person Schema:");
    println!("-----------------");
    let person_schema = Person::schema();
    println!("Name: {}", person_schema.name);
    if let Some(desc) = &person_schema.description {
        println!("Description: {}", desc);
    }
    println!(
        "Schema: {}",
        serde_json::to_string_pretty(&person_schema.schema).unwrap()
    );

    // Example 2: Product schema
    println!("\n\n2. Product Schema:");
    println!("------------------");
    let product_schema = Product::schema();
    println!("Name: {}", product_schema.name);
    println!(
        "Schema: {}",
        serde_json::to_string_pretty(&product_schema.schema).unwrap()
    );

    // Example 3: Complex nested schema
    println!("\n\n3. Order Schema (nested):");
    println!("--------------------------");
    let order_schema = Order::schema();
    println!("Name: {}", order_schema.name);
    if let Some(desc) = &order_schema.description {
        println!("Description: {}", desc);
    }
    println!(
        "Schema: {}",
        serde_json::to_string_pretty(&order_schema.schema).unwrap()
    );

    // Example 4: Demonstrate JSON validation
    println!("\n\n4. Example JSON that matches schemas:");
    println!("-------------------------------------");

    let person_json = serde_json::json!({
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    });
    println!(
        "Valid Person JSON: {}",
        serde_json::to_string_pretty(&person_json).unwrap()
    );

    let order_json = serde_json::json!({
        "order_id": "ORD-12345",
        "customer": {
            "name": "Jane Smith",
            "age": 28,
            "email": "jane@example.com"
        },
        "items": [
            {
                "product_id": "PROD-001",
                "quantity": 2,
                "price": 29.99
            },
            {
                "product_id": "PROD-002",
                "quantity": 1,
                "price": 49.99
            }
        ],
        "total": 109.97,
        "status": "processing"
    });
    println!(
        "\nValid Order JSON: {}",
        serde_json::to_string_pretty(&order_json).unwrap()
    );

    // Example 5: Deserialize and validate
    println!("\n\n5. Deserialization Test:");
    println!("------------------------");

    match serde_json::from_value::<Person>(person_json) {
        Ok(person) => println!("✓ Successfully deserialized Person: {:?}", person),
        Err(e) => println!("✗ Failed to deserialize Person: {}", e),
    }

    match serde_json::from_value::<Order>(order_json) {
        Ok(order) => println!("✓ Successfully deserialized Order: {:?}", order),
        Err(e) => println!("✗ Failed to deserialize Order: {}", e),
    }

    println!("\n\nNote: In a real agent, these schemas would be sent to the LLM");
    println!("to ensure structured, type-safe responses that can be parsed automatically.");
}
