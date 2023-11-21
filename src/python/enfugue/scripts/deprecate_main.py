def deprecate_main() -> None:
    """
    Indicate the console script is decprecated.
    """
    import termcolor
    print(termcolor.colored("Executing `enfugue` directly is deprecated. Use `python3 -m enfugue <command>` instead.", "yellow"))

if __name__ == "__main__":
    deprecate_main()
