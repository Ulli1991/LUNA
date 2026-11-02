"""ASCII LUNA startup banner."""


def print_LUNA_banner():
    """Print the startup banner."""
    print("Hi there, this is LUNA:\n")
    print(
        "A differentiable astrophysical hydrodynamics code for turbulence, shocks, ISM, cosmology, and galaxy evolution\n"
    )

    L = ["#      ", "#      ", "#      ", "#      ", "#      ", "#      ", "#######"]

    U = ["#     #", "#     #", "#     #", "#     #", "#     #", "#     #", " ##### "]

    N = ["#     #", "##    #", "# #   #", "#  #  #", "#   # #", "#    ##", "#     #"]

    A = ["   #   ", "  # #  ", " #   # ", "#     #", "#######", "#     #", "#     #"]

    for row in range(len(L)):
        line = L[row] + " " + U[row] + " " + N[row] + " " + A[row]
        print(line)

    print("\nThis is version 0.1.0.\n")


if __name__ == "__main__":
    print_LUNA_banner()
