import auth.userAuth as userAuth

# core application logic
def main():
    test = userAuth.UserAuth()
    print("Starting the application...")
    if test.test_authentication() == True:
        print("Authentication successful!")
    else :
        print("Authentication failed.")

main()