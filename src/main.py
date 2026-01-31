import auth.userAuth as userAuth


def main():
    test = userAuth.UserAuth()
    print("Starting the application...")
    if test.test_authentication() == True:
        print("Authentication successful!")
    else :
        print("Authentication failed.")

main()