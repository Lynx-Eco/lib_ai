use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lit, Meta};

/// Derive macro for StructuredProvider trait
///
/// This macro automatically generates a JSON schema for a struct,
/// making it usable with structured output in AI agents.
///
/// # Example
/// ```
/// use lib_ai_derive::Structured;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, Structured)]
/// struct WeatherResponse {
///     #[schema(description = "Current temperature in Celsius")]
///     temperature: f32,
///     
///     #[schema(description = "Weather condition")]
///     condition: String,
///     
///     #[schema(description = "Humidity percentage")]
///     humidity: u8,
/// }
/// ```
#[proc_macro_derive(Structured, attributes(schema))]
pub fn derive_structured(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let schema_json = generate_schema(&input);

    let name_str = name.to_string();

    let expanded = quote! {
        impl lib_ai::agent::StructuredProvider for #name {
            fn schema() -> lib_ai::JsonSchema {
                lib_ai::JsonSchema {
                    name: #name_str.to_string(),
                    description: None,
                    schema: #schema_json,
                    strict: Some(true),
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for creating tool executors
///
/// This macro generates a ToolExecutor implementation for a struct,
/// allowing it to be used as a tool in AI agents.
///
/// # Example
/// ```
/// use lib_ai_derive::AiTool;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, Serialize, Deserialize, AiTool)]
/// #[tool(name = "weather", description = "Get current weather information")]
/// struct WeatherTool {
///     #[tool(description = "City name")]
///     city: String,
///     
///     #[tool(description = "Temperature unit", enum_values = "celsius,fahrenheit")]
///     unit: String,
/// }
///
/// impl WeatherTool {
///     // This method will be called when the tool is executed
///     async fn execute(self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
///         // Implementation here
///         Ok(serde_json::json!({
///             "temperature": 22.5,
///             "condition": "sunny",
///             "city": self.city,
///             "unit": self.unit
///         }))
///     }
/// }
/// ```
#[proc_macro_derive(AiTool, attributes(tool))]
pub fn derive_ai_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let (tool_name, tool_description) = extract_tool_attributes(&input);
    let parameters_schema = generate_tool_parameters(&input);

    let expanded = quote! {
        #[async_trait::async_trait]
        impl lib_ai::agent::ToolExecutor for #name {
            async fn execute(&self, arguments: &str) -> Result<lib_ai::agent::ToolResult, Box<dyn std::error::Error>> {
                let tool_input: #name = serde_json::from_str(arguments)?;
                match tool_input.execute().await {
                    Ok(result) => Ok(lib_ai::agent::ToolResult::Success(result)),
                    Err(e) => Ok(lib_ai::agent::ToolResult::Error(e.to_string())),
                }
            }

            fn definition(&self) -> lib_ai::ToolFunction {
                lib_ai::ToolFunction {
                    name: #tool_name.to_string(),
                    description: Some(#tool_description.to_string()),
                    parameters: #parameters_schema,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_schema(input: &DeriveInput) -> proc_macro2::TokenStream {
    match &input.data {
        Data::Struct(data_struct) => {
            let properties = generate_struct_properties(&data_struct.fields);
            let required = generate_required_fields(&data_struct.fields);

            quote! {
                serde_json::json!({
                    "type": "object",
                    "properties": #properties,
                    "required": #required
                })
            }
        }
        Data::Enum(data_enum) => {
            let variants: Vec<_> = data_enum
                .variants
                .iter()
                .map(|v| {
                    let name = v.ident.to_string();
                    quote! { #name }
                })
                .collect();

            quote! {
                serde_json::json!({
                    "type": "string",
                    "enum": [#(#variants),*]
                })
            }
        }
        _ => panic!("Structured can only be derived for structs and enums"),
    }
}

fn generate_struct_properties(fields: &Fields) -> proc_macro2::TokenStream {
    match fields {
        Fields::Named(fields) => {
            let field_schemas: Vec<proc_macro2::TokenStream> = fields
                .named
                .iter()
                .map(|f| {
                    let field_name = f.ident.as_ref().unwrap().to_string();
                    let field_type = &f.ty;
                    let description = extract_description(&f.attrs);

                    let type_str = match quote!(#field_type).to_string().as_str() {
                        "String" => "string",
                        "bool" => "boolean",
                        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "integer",
                        "f32" | "f64" => "number",
                        _ => "object", // Default to object for complex types
                    };

                    if let Some(desc) = description {
                        quote! {
                            #field_name: {
                                "type": #type_str,
                                "description": #desc
                            }
                        }
                    } else {
                        quote! {
                            #field_name: {
                                "type": #type_str
                            }
                        }
                    }
                })
                .collect();

            quote! {
                serde_json::json!({
                    #(#field_schemas),*
                })
            }
        }
        _ => panic!("Only named fields are supported"),
    }
}

fn generate_required_fields(fields: &Fields) -> proc_macro2::TokenStream {
    match fields {
        Fields::Named(fields) => {
            let required: Vec<_> = fields
                .named
                .iter()
                .filter(|f| !is_option_type(&f.ty))
                .map(|f| {
                    let name = f.ident.as_ref().unwrap().to_string();
                    quote! { #name }
                })
                .collect();

            quote! {
                serde_json::json!([#(#required),*])
            }
        }
        _ => quote! { serde_json::json!([]) },
    }
}

fn generate_tool_parameters(input: &DeriveInput) -> proc_macro2::TokenStream {
    match &input.data {
        Data::Struct(data_struct) => {
            let properties = generate_tool_properties(&data_struct.fields);
            let required = generate_required_fields(&data_struct.fields);

            quote! {
                serde_json::json!({
                    "type": "object",
                    "properties": #properties,
                    "required": #required
                })
            }
        }
        _ => panic!("AiTool can only be derived for structs"),
    }
}

fn generate_tool_properties(fields: &Fields) -> proc_macro2::TokenStream {
    match fields {
        Fields::Named(fields) => {
            let field_schemas: Vec<proc_macro2::TokenStream> = fields
                .named
                .iter()
                .map(|f| {
                    let field_name = f.ident.as_ref().unwrap().to_string();
                    let field_type = &f.ty;
                    let attrs = extract_tool_field_attributes(&f.attrs);

                    let type_str = match quote!(#field_type).to_string().as_str() {
                        "String" => "string",
                        "bool" => "boolean",
                        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "integer",
                        "f32" | "f64" => "number",
                        _ => "object",
                    };

                    let mut field_def = vec![quote! { "type": #type_str }];

                    if let Some(desc) = attrs.description {
                        field_def.push(quote! { "description": #desc });
                    }

                    if let Some(enum_values) = attrs.enum_values {
                        let values: Vec<_> =
                            enum_values.split(',').map(|v| quote! { #v }).collect();
                        field_def.push(quote! { "enum": [#(#values),*] });
                    }

                    quote! {
                        #field_name: {
                            #(#field_def),*
                        }
                    }
                })
                .collect();

            quote! {
                serde_json::json!({
                    #(#field_schemas),*
                })
            }
        }
        _ => panic!("Only named fields are supported"),
    }
}

struct ToolFieldAttributes {
    description: Option<String>,
    enum_values: Option<String>,
}

fn extract_tool_field_attributes(attrs: &[syn::Attribute]) -> ToolFieldAttributes {
    let mut result = ToolFieldAttributes {
        description: None,
        enum_values: None,
    };

    for attr in attrs {
        if attr.path().is_ident("tool") {
            let meta_list = match &attr.meta {
                Meta::List(list) => list,
                _ => continue,
            };

            let parsed = meta_list.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            );

            if let Ok(nested) = parsed {
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) if nv.path.is_ident("description") => {
                            if let syn::Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    result.description = Some(lit_str.value());
                                }
                            }
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("enum_values") => {
                            if let syn::Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    result.enum_values = Some(lit_str.value());
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    result
}

fn extract_description(attrs: &[syn::Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("schema") {
            if let Meta::List(meta_list) = &attr.meta {
                if let Ok(Meta::NameValue(nv)) = meta_list.parse_args::<Meta>() {
                    if nv.path.is_ident("description") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let Lit::Str(lit_str) = &expr_lit.lit {
                                return Some(lit_str.value());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn extract_tool_attributes(input: &DeriveInput) -> (String, String) {
    let mut name = input.ident.to_string();
    let mut description = format!("{} tool", name);

    for attr in &input.attrs {
        if attr.path().is_ident("tool") {
            let meta_list = match &attr.meta {
                Meta::List(list) => list,
                _ => continue,
            };

            let parsed = meta_list.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            );

            if let Ok(nested) = parsed {
                for meta in nested {
                    match meta {
                        Meta::NameValue(nv) if nv.path.is_ident("name") => {
                            if let syn::Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    name = lit_str.value();
                                }
                            }
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("description") => {
                            if let syn::Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    description = lit_str.value();
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    (name, description)
}

fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            return segment.ident == "Option";
        }
    }
    false
}
