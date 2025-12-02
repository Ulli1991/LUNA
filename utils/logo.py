def print_KRATOS_banner():
    # Intro line
    print("Hi there, this is KRATOS:\n")
    print("A differentiable astrophysical MHD code for turbulence, shocks, ISM, and galaxy evolution\n")

    # ASCII letters
    K = [
        "#    # ",
        "#   #  ",
        "#  #   ",
        "###    ",
        "#  #   ",
        "#   #  ",
        "#    # "
    ]

    R = [
        "###### ",
        "#     #",
        "#     #",
        "###### ",
        "#  #   ",
        "#   #  ",
        "#    # "
    ]

    A = [
        "   #   ",
        "  # #  ",
        " #   # ",
        "#     #",
        "#######",
        "#     #",
        "#     #"
    ]

    T = [
        "#######",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   "
    ]

    O = [
        " ##### ",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        " ##### "
    ]

    S = [
        " ##### ",
        "#     #",
        "#      ",
        " ##### ",
        "      #",
        "#     #",
        " ##### "
    ]

    # Combine letters row by row
    for row in range(len(K)):
        line = K[row] + " " + R[row] + " " + A[row] + " " + T[row] + " " + O[row] + " " + S[row]
        print(line)

    # Bottom line
    print("\nThis is version 0.1.\n")

if __name__ == "__main__":
    print_KRATOS_banner()
