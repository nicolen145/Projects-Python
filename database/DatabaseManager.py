import pyodbc
import csv

class DatabaseManager:
    def __init__(self, driver: str, server: str, username: str, password: str):
        """
        Initialize the DatabaseManager with a connection to the SQL Server database.

        :param driver: The ODBC driver name.
        :param server: The server name or IP address.
        :param username: The database username.
        :param password: The database password.
        """
        # Connect to the database using pyodbc
        self.connection = pyodbc.connect(
            f'DRIVER={driver};SERVER={server};DATABASE={username};UID={username};PWD={password}'
        )

    def file_to_database(self, path: str) -> None:
        """
        Load data from a CSV file into the MediaItems table.
        :param path: Path to the CSV file containing TITLE and PROD_YEAR columns.
        """
        cursor = self.connection.cursor()
        with open(path, 'r') as file:
            reader = csv.reader(file)
            for title, prod_year in reader:
                # Insert each row from the CSV into the MediaItems table
                cursor.execute(
                    "INSERT INTO MediaItems (TITLE, PROD_YEAR) VALUES (?, ?)", title, prod_year
                )
        # Commit the transaction to save changes
        self.connection.commit()

    def calculate_similarity(self) -> None:
        """
        Calculate similarity scores between media items and store them in the Similarity table.
        """
        cursor = self.connection.cursor()
        # Get the maximal distance using the dbo.MaximalDistance() function
        cursor.execute("SELECT dbo.MaximalDistance()")
        maximal_distance = cursor.fetchone()[0]
        # Fetch all Media IDs from the MediaItems table
        cursor.execute("SELECT MID FROM MediaItems")
        media_ids = [row[0] for row in cursor.fetchall()]
        # Calculate similarity between each pair of media items
        for i, mid1 in enumerate(media_ids):
            for mid2 in media_ids[i+1:]:
                # Call dbo.SimCalculation() to compute the similarity
                cursor.execute(
                    "SELECT dbo.SimCalculation(?, ?, ?)", mid1, mid2, maximal_distance
                )
                similarity = cursor.fetchone()[0]
                # Insert the similarity score into the Similarity table
                cursor.execute(
                    "INSERT INTO Similarity (MID1, MID2, SIMILARITY) VALUES (?, ?, ?)",
                    mid1, mid2, similarity
                )
        # Commit the transaction to save changes
        self.connection.commit()

    def print_similar_items(self, mid: int) -> None:
        """
        Print similar media items for a given media ID with a similarity score of at least 0.25.
        :param mid: The Media ID to find similar items for.
        """
        cursor = self.connection.cursor()
        # Query to fetch similar items and their similarity scores from both directions
        cursor.execute("""
            SELECT m.TITLE, s.SIMILARITY
        FROM Similarity s
        JOIN MediaItems m ON (s.MID2 = m.MID AND s.MID1 = ?) OR (s.MID1 = m.MID AND s.MID2 = ?)
        WHERE s.SIMILARITY >= 0.25
        ORDER BY s.SIMILARITY ASC
        """, (mid, mid))
        # Print the results
        for title, similarity in cursor.fetchall():
            print(f"{title}: {similarity}")

    def add_summary_items(self) -> None:
        """
        Execute the stored procedure AddSummaryItems to add summary data to the database.
        """
        cursor = self.connection.cursor()
        # Call the stored procedure AddSummaryItems
        cursor.execute("EXEC AddSummaryItems")
        # Commit the transaction to save changes
        self.connection.commit()
