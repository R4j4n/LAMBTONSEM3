import os
import sqlite3

# Define the database file path
db_file = "petshop.db"

# Remove the database file if it already exists
if os.path.exists(db_file):
    os.remove(db_file)

# Connect to the SQLite database (this will create the file if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# The SQL schema script
schema_sql = """
-- Pet Shop Database Schema

-- Products table for storing pet shop inventory
CREATE TABLE Products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0,
    supplier TEXT,
    description TEXT,
    added_date TEXT DEFAULT (datetime('now', 'localtime'))
);

-- Customers table for storing customer information
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    address TEXT,
    join_date TEXT DEFAULT (datetime('now', 'localtime')),
    loyalty_points INTEGER DEFAULT 0
);

-- Sales table for tracking transactions
CREATE TABLE Sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    sale_date TEXT DEFAULT (datetime('now', 'localtime')),
    total_amount REAL NOT NULL,
    payment_method TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

-- SaleItems table for items within each sale
CREATE TABLE SaleItems (
    sale_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sale_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    price_each REAL NOT NULL,
    subtotal REAL NOT NULL,
    FOREIGN KEY (sale_id) REFERENCES Sales(sale_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

-- Pets table for animals available for sale
CREATE TABLE Pets (
    pet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    species TEXT NOT NULL,
    breed TEXT,
    age INTEGER,
    gender TEXT CHECK(gender IN ('Male', 'Female')),
    color TEXT,
    price REAL NOT NULL,
    arrival_date TEXT DEFAULT (datetime('now', 'localtime')),
    health_status TEXT DEFAULT 'Healthy',
    sold INTEGER DEFAULT 0 CHECK(sold IN (0, 1))
);

-- Services table for grooming, vet visits, etc.
CREATE TABLE Services (
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    duration INTEGER, -- in minutes
    price REAL NOT NULL
);

-- Appointments table for service bookings
CREATE TABLE Appointments (
    appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    service_id INTEGER NOT NULL,
    pet_id INTEGER,
    appointment_date TEXT NOT NULL,
    status TEXT DEFAULT 'Scheduled' CHECK(status IN ('Scheduled', 'Completed', 'Cancelled')),
    notes TEXT,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (service_id) REFERENCES Services(service_id),
    FOREIGN KEY (pet_id) REFERENCES Pets(pet_id)
);

-- Employee table
CREATE TABLE Employees (
    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    position TEXT NOT NULL,
    hire_date TEXT DEFAULT (datetime('now', 'localtime')),
    email TEXT UNIQUE,
    phone TEXT,
    salary REAL
);

-- Insert sample data into Products
INSERT INTO Products (name, category, price, stock_quantity, supplier, description) VALUES
('Premium Dog Food', 'Food', 29.99, 50, 'PetNutrition Inc.', 'High-quality dog food for all breeds'),
('Cat Scratching Post', 'Accessories', 19.99, 25, 'PetFurniture Co.', 'Durable cat scratching post with toy'),
('Bird Cage - Medium', 'Housing', 49.99, 10, 'WildLife Supplies', 'Medium-sized bird cage suitable for parakeets'),
('Fish Tank Filter', 'Equipment', 24.99, 30, 'AquaLife', 'Water filter for fish tanks up to 20 gallons'),
('Dog Leash', 'Accessories', 14.99, 40, 'PetWalk', 'Sturdy nylon dog leash, 6 feet'),
('Hamster Wheel', 'Accessories', 9.99, 20, 'Small Pets Inc.', 'Exercise wheel for hamsters and small rodents'),
('Cat Litter - 10lb', 'Supplies', 12.99, 60, 'CleanPets', 'Clumping cat litter, odor control'),
('Dog Chew Toy', 'Toys', 8.99, 45, 'PlayPets', 'Durable rubber chew toy for dogs');

-- Insert sample data into Customers
INSERT INTO Customers (first_name, last_name, email, phone, address, loyalty_points) VALUES
('John', 'Smith', 'john.smith@email.com', '555-123-4567', '123 Main St, Anytown, USA', 150),
('Sarah', 'Johnson', 'sarah.j@email.com', '555-234-5678', '456 Oak Ave, Somewhere, USA', 75),
('Michael', 'Williams', 'michael.w@email.com', '555-345-6789', '789 Pine Rd, Nowhere, USA', 200),
('Emily', 'Brown', 'emily.b@email.com', '555-456-7890', '321 Maple Dr, Everywhere, USA', 50),
('David', 'Jones', 'david.j@email.com', '555-567-8901', '654 Elm Blvd, Anywhere, USA', 125);

-- Insert sample data into Pets
INSERT INTO Pets (species, breed, age, gender, color, price, health_status) VALUES
('Dog', 'Labrador Retriever', 6, 'Male', 'Yellow', 599.99, 'Healthy'),
('Cat', 'Siamese', 12, 'Female', 'Cream', 349.99, 'Healthy'),
('Bird', 'Parakeet', 8, 'Male', 'Blue', 49.99, 'Healthy'),
('Rabbit', 'Dutch', 5, 'Female', 'Black and White', 79.99, 'Healthy'),
('Hamster', 'Syrian', 2, 'Male', 'Golden', 24.99, 'Healthy'),
('Fish', 'Goldfish', 6, 'Male', 'Orange', 9.99, 'Healthy'),
('Dog', 'Beagle', 9, 'Female', 'Tri-color', 499.99, 'Healthy');

-- Insert sample data into Services
INSERT INTO Services (name, description, duration, price) VALUES
('Basic Grooming', 'Bath, brush, and nail trim', 60, 45.99),
('Full Grooming', 'Bath, brush, nail trim, haircut, ear cleaning', 120, 89.99),
('Veterinary Check-up', 'Basic health examination', 30, 59.99),
('Pet Training - Basic', 'Introductory obedience training session', 45, 39.99),
('Pet Training - Advanced', 'Advanced commands and behavior training', 60, 59.99);

-- Insert sample data into Employees
INSERT INTO Employees (first_name, last_name, position, email, phone, salary) VALUES
('Robert', 'Anderson', 'Manager', 'robert.a@petshop.com', '555-111-2222', 58000),
('Jessica', 'Martinez', 'Veterinarian', 'jessica.m@petshop.com', '555-222-3333', 75000),
('Thomas', 'Garcia', 'Groomer', 'thomas.g@petshop.com', '555-333-4444', 42000),
('Amanda', 'Lee', 'Sales Associate', 'amanda.l@petshop.com', '555-444-5555', 32000),
('Kevin', 'Wilson', 'Pet Trainer', 'kevin.w@petshop.com', '555-555-6666', 38000);

-- Insert sample data into Sales
INSERT INTO Sales (customer_id, sale_date, total_amount, payment_method) VALUES
(1, datetime('now', '-10 days'), 54.97, 'Credit Card'),
(3, datetime('now', '-7 days'), 89.99, 'Credit Card'),
(2, datetime('now', '-5 days'), 12.99, 'Cash'),
(4, datetime('now', '-3 days'), 34.98, 'Debit Card'),
(5, datetime('now', '-1 day'), 49.99, 'Credit Card');

-- Insert sample data into SaleItems
INSERT INTO SaleItems (sale_id, product_id, quantity, price_each, subtotal) VALUES
(1, 1, 1, 29.99, 29.99),
(1, 5, 1, 14.99, 14.99),
(1, 8, 1, 8.99, 8.99),
(2, 3, 1, 49.99, 49.99),
(2, 4, 1, 24.99, 24.99),
(2, 6, 1, 9.99, 9.99),
(2, 8, 1, 8.99, 8.99),
(3, 7, 1, 12.99, 12.99),
(4, 5, 1, 14.99, 14.99),
(4, 8, 2, 8.99, 17.98),
(5, 3, 1, 49.99, 49.99);

-- Insert sample data into Appointments
INSERT INTO Appointments (customer_id, service_id, pet_id, appointment_date, status) VALUES
(1, 1, 1, datetime('now', '+2 days'), 'Scheduled'),
(2, 2, 2, datetime('now', '+3 days'), 'Scheduled'),
(3, 3, NULL, datetime('now', '+1 day'), 'Scheduled'),
(4, 4, 4, datetime('now', '-5 days'), 'Completed'),
(5, 5, NULL, datetime('now', '-2 days'), 'Cancelled');

-- Create triggers

-- Trigger to update stock quantity when a sale is made
CREATE TRIGGER after_sale_item_insert
AFTER INSERT ON SaleItems
BEGIN
    UPDATE Products
    SET stock_quantity = stock_quantity - NEW.quantity
    WHERE product_id = NEW.product_id;
END;

-- Trigger to update loyalty points when a sale is made
CREATE TRIGGER after_sale_insert
AFTER INSERT ON Sales
BEGIN
    UPDATE Customers
    SET loyalty_points = loyalty_points + CAST(NEW.total_amount AS INTEGER)
    WHERE customer_id = NEW.customer_id;
END;
"""

# Execute the schema script
cursor.executescript(schema_sql)

# Commit the changes and close the connection
conn.commit()
print(f"Database '{db_file}' created successfully with all tables and sample data.")


# Verify the database by querying some data
def verify_database():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nDatabase tables:")
    for table in tables:
        print(f"- {table[0]}")

    # Check product count
    cursor.execute("SELECT COUNT(*) FROM Products;")
    product_count = cursor.fetchone()[0]
    print(f"\nNumber of products: {product_count}")

    # Check upcoming appointments with a direct query (no view)
    cursor.execute(
        """
    SELECT
        a.appointment_id,
        a.appointment_date,
        s.name AS service_name,
        c.first_name || ' ' || c.last_name AS customer_name
    FROM Appointments a
    JOIN Customers c ON a.customer_id = c.customer_id
    JOIN Services s ON a.service_id = s.service_id
    WHERE a.status = 'Scheduled' AND a.appointment_date >= datetime('now', 'localtime')
    ORDER BY a.appointment_date
    """
    )
    appointments = cursor.fetchall()
    print(f"\nUpcoming appointments: {len(appointments)}")
    for appt in appointments:
        print(
            f"- ID: {appt[0]}, Date: {appt[1]}, Service: {appt[2]}, Customer: {appt[3]}"
        )

    # Check products with low stock (direct query instead of view)
    cursor.execute(
        """
    SELECT
        product_id,
        name,
        category,
        stock_quantity
    FROM Products
    WHERE stock_quantity < 15
    """
    )
    low_stock = cursor.fetchall()
    print(f"\nProducts with low stock: {len(low_stock)}")
    for prod in low_stock:
        print(
            f"- ID: {prod[0]}, Name: {prod[1]}, Category: {prod[2]}, Stock: {prod[3]}"
        )

    conn.close()
    print("\nVerification complete!")


# Run verification
verify_database()
