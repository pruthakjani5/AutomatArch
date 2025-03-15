
# Automata Architect: Regular Expression to Minimized DFA Converter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automatarch.streamlit.app/)

Automata Architect is a web application that allows you to convert regular expressions into minimized Deterministic Finite Automata (DFA). It provides a comprehensive visualization and explanation of the conversion process, making it an excellent tool for learning and experimentation with automata theory.

**Live Demo:** [https://automatarch.streamlit.app/](https://automatarch.streamlit.app/)

## Features

-   **Regular Expression to NFA-ε Conversion:** Converts your regular expression into a Non-deterministic Finite Automaton with epsilon transitions (NFA-ε).
-   **NFA-ε to DFA Conversion:** Transforms the NFA-ε into a Deterministic Finite Automaton (DFA) using the subset construction method.
-   **DFA Minimization:** Minimizes the DFA to its simplest form, reducing the number of states while preserving the language it accepts.
-   **Visualization:** Provides graphical visualizations of the NFA-ε, DFA, and minimized DFA using Graphviz.
-   **Transition Tables:** Displays the transition tables for the DFA and minimized DFA, clearly showing the state transitions.
-   **Explanations:** Offers detailed explanations of the NFA to DFA conversion process and the language defined by the regular expression.
-   **String Testing:** Allows you to test strings against the minimized DFA to see if they are accepted.
-   **User-Friendly Interface:** Built with Streamlit, providing an intuitive and interactive web interface.

## How to Use

1.  **Enter Regular Expression:** Input your regular expression in the text box.
2.  **View Results:** The application will process your input and display the NFA-ε, DFA, and minimized DFA, along with their visualizations and transition tables.
3.  **Explore Explanations:** The sidebar provides detailed explanations of the conversion process and the regular expression language.
4.  **Test Strings:** Use the string testing feature to check if your strings are accepted by the minimized DFA.

## Deployment

This application is deployed on Streamlit Cloud. To deploy your own instance:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pruthakjani5/automatarch
    cd automatarch
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Locally (Optional):**
    ```bash
    streamlit run app.py
    ```

**`requirements.txt`:**

```
streamlit
graphviz
pandas
```

**`packages.txt`:**

```
graphviz
```

## Contributing

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

## Author

This application was created by Pruthak Jani.

**Connect with Me:** If you have any questions, feedback, bug reports, or if you find any incorrect answers, please feel free to reach out via email: pruthak.jani@gmail.com

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.
