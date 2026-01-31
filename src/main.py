import auth.user_auth as user_auth
import data.overview.all_markets as all_markets

# core application logic
def main():
    test = user_auth.UserAuth()
    print("Starting the application...")
    if test.test_authentication() == True:
        print("Authentication successful!")
        print("Generating overview file and saving snapshot to DB...")
        overview = all_markets.OverviewAllMarkets()
        # Still write the JSON file locally (optional)
        # overview.create_overview_file()
        # Also persist the same snapshot into Postgres
        overview.save_overview_to_db()
    else :
        print("Authentication failed.")

    

main()