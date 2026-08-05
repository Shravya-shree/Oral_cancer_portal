# update_db.py
import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1239",  # Your MySQL password
        database="oral_cancer_db"
    )
    cursor = conn.cursor()
    
    # Add the gradcam_image column directly to the scans table
    query = "ALTER TABLE scans ADD COLUMN gradcam_image VARCHAR(255) AFTER image;"
    cursor.execute(query)
    conn.commit()
    
    print("\n✅ SUCCESS: 'gradcam_image' column added successfully to 'scans' table!\n")

except mysql.connector.Error as err:
    if err.errno == 1060:  # Code 1060 = Duplicate column name error
        print("\nℹ️ NOTICE: Column 'gradcam_image' already exists in table 'scans'. You are good to go!\n")
    else:
        print(f"\n❌ Error connecting or updating database: {err}\n")

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()