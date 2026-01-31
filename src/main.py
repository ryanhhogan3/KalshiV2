import auth.user_auth as user_auth
import data.overview.all_markets as all_markets

# core application logic
def main():
    test = user_auth.UserAuth()
    print("Starting the application...")
    if test.test_authentication() == True:
        print("Authentication successful!")
        print("Generating overview file...")
        overview = all_markets.OverviewAllMarkets()
        overview.create_overview_file()
    else :
        print("Authentication failed.")

    

main()