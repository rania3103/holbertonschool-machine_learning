-- a script that creates a table users
CREATE TABLE IF NOT EXISTS users(
id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
email varchar(255) NOT NULL UNIQUE,
name varchar(255),
country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'
);
