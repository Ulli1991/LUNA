def print_LUNA_banner():
    # Intro line
    print("Hi there, this is LUNA:\n")
    print(
        "A differentiable astrophysical MHD code for turbulence, shocks, ISM, and galaxy evolution\n"
    )

    # ASCII letters
    L = ["#      ", "#      ", "#      ", "#      ", "#      ", "#      ", "#######"]

    U = ["#     #", "#     #", "#     #", "#     #", "#     #", "#     #", " ##### "]

    N = ["#     #", "##    #", "# #   #", "#  #  #", "#   # #", "#    ##", "#     #"]

    A = ["   #   ", "  # #  ", " #   # ", "#     #", "#######", "#     #", "#     #"]

    # Combine letters row by row
    for row in range(len(L)):
        line = L[row] + " " + U[row] + " " + N[row] + " " + A[row]
        print(line)

    # Bottom line
    print("\nThis is version 0.1.\n")


if __name__ == "__main__":
    print_LUNA_banner()
