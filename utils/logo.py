def print_SPITE_banner():
    # Intro line
    print("Hi there, this is the Simulation Platform for Interstellar, Turbulent & Extragalactic physics, but you can call me:\n")

    # ASCII letters
    S = [
        "  #####",
        " #     ",
        "#      ",
        " ####  ",
        "      #",
        "      #",
        " ##### "
    ]

    P = [
        "###### ",
        "#     #",
        "#     #",
        "###### ",
        "#      ",
        "#      ",
        "#      "
    ]

    I = [
        "#######",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   ",
        "   #   ",
        "#######"
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

    E = [
        "#######",
        "#      ",
        "#      ",
        "#####  ",
        "#      ",
        "#      ",
        "#######"
    ]

    # Combine letters row by row
    for row in range(len(S)):
        line = S[row] + " " + P[row] + " " + I[row] + " " + T[row] + " " + E[row]
        print(line)

    # Bottom line
    print("This is version 0.1.\n")

if __name__ == "__main__":
    print_SPITE_banner()
