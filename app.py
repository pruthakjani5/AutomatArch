import streamlit as st
import itertools
from collections import defaultdict
import graphviz
import re
import pandas as pd

class State:
    """Represents a state in the automaton."""
    def __init__(self, name):
        self.name = name
        
    def __repr__(self):
        return self.name

class NFA:
    """Represents an NFA with states, alphabet, transitions, start and final states."""
    def __init__(self, states, alphabet, transitions, start_state, final_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.final_states = final_states

    def epsilon_closure(self, state):
        """Compute the epsilon closure of a given state."""
        stack = [state]
        closure = set(stack)

        while stack:
            current_state = stack.pop()
            if (current_state, '') in self.transitions:
                for next_state in self.transitions[(current_state, '')]:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
        return closure

    def to_dfa(self):
        """Convert an NFA to a DFA using subset construction and handle dead states."""
       	initial_closure = frozenset(self.epsilon_closure(self.start_state))
        dfa_states = {initial_closure}
        dfa_transitions = {}
        unprocessed_states = [initial_closure]
        state_mapping = {initial_closure: f'q0'}
        dfa_final_states = set()
        state_counter = itertools.count(1)

        # Check if the initial state contains any final states
        for nfa_state in initial_closure:
            if nfa_state in self.final_states:
                dfa_final_states.add('q0')
                break

        while unprocessed_states:
            current_dfa_state = unprocessed_states.pop(0)

            for symbol in self.alphabet:
                if symbol == '':
                    continue  # Skip epsilon

                next_state_set = set()
                for nfa_state in current_dfa_state:
                    if (nfa_state, symbol) in self.transitions:
                        for next_nfa_state in self.transitions[(nfa_state, symbol)]:
                            next_state_set.update(self.epsilon_closure(next_nfa_state))

                next_state_frozen = frozenset(next_state_set)
                if next_state_set:
                    if next_state_frozen not in dfa_states:
                        dfa_states.add(next_state_frozen)
                        unprocessed_states.append(next_state_frozen)
                        state_mapping[next_state_frozen] = f'q{next(state_counter)}'
                        
                        # Check if the new state contains any final states
                        for nfa_state in next_state_frozen:
                            if nfa_state in self.final_states:
                                dfa_final_states.add(state_mapping[next_state_frozen])
                                break

                    dfa_transitions[(state_mapping[current_dfa_state], symbol)] = state_mapping[next_state_frozen]

        # Identify and handle dead state if any symbol is missing in DFA
        dead_state = f'q{next(state_counter)}'
        added_dead_state = False
        all_symbols = set(filter(lambda x: x != '', self.alphabet))

        for state in state_mapping.values():
            for symbol in all_symbols:
                if (state, symbol) not in dfa_transitions:
                    dfa_transitions[(state, symbol)] = dead_state
                    added_dead_state = True

        # Only add the dead state if it's actually used in transitions
        if added_dead_state:
            # Add transitions from dead state to itself for all symbols
            for symbol in all_symbols:
                dfa_transitions[(dead_state, symbol)] = dead_state
            dfa_states = set(state_mapping.values()) | {dead_state}
        else:
            dfa_states = set(state_mapping.values())
	
        start_dfa_state = state_mapping[initial_closure]
        for symbol in self.alphabet:
            if symbol == '': continue
            if (start_dfa_state, symbol) not in dfa_transitions: dfa_transitions[(start_dfa_state, symbol)] = start_dfa_state

        return {
            "states": dfa_states,
            "alphabet": all_symbols,
            "transitions": dfa_transitions,
            "start_state": state_mapping[initial_closure],
            "final_states": dfa_final_states,
        }

    def display(self):
        """Display the NFA details."""
        print("\nNFA Details:")
        print("States:", self.states)
        print("Alphabet:", self.alphabet)
        print("Start State:", self.start_state)
        print("Final States:", self.final_states)
        print("Transitions:")
        for (state, symbol), next_states in self.transitions.items():
            print(f"δ({state}, {symbol if symbol else 'ε'}) → {next_states}")

def precedence(op):
    """Defines operator precedence."""
    if op == '*':
        return 3
    elif op == '.':
        return 2
    elif op == '+':  # Union operator
        return 1
    return 0

def insert_concat_operators(regex):
    """Insert explicit concatenation operators (.) where necessary."""
    if not regex:
        return regex
        
    output = []
    for i in range(len(regex) - 1):
        output.append(regex[i])

        # Insert `.` for explicit concatenation in cases like ab -> a.b
        if (
            (regex[i].isalnum() or regex[i] == ')' or regex[i] == '*') and
            (regex[i+1].isalnum() or regex[i+1] == '(')
        ):
            output.append('.')
    
    output.append(regex[-1])  # Append last character
    return ''.join(output)

def infix_to_postfix(regex):
    """Convert infix regular expression to postfix (RPN)."""
    regex = insert_concat_operators(regex)
    output = []
    stack = []
    
    for char in regex:
        if char.isalnum():
            output.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack and stack[-1] == '(':
                stack.pop()  # Discard the '('
        else:  # Operator
            while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(char):
                output.append(stack.pop())
            stack.append(char)

    while stack:
        output.append(stack.pop())

    return ''.join(output)

def re_to_nfa(re):
    """Convert a regular expression to an ε-NFA using Thompson's Construction."""
    if not re:
        return None
        
    postfix_re = infix_to_postfix(re)  # Convert to postfix
    print(f"Postfix notation: {postfix_re}")
    stack = []
    state_counter = itertools.count()
    
    def new_state():
        return f'q{next(state_counter)}'

    for char in postfix_re:
        if char.isalnum():  # Single character automaton
            start, end = new_state(), new_state()
            transitions = {(start, char): {end}}
            stack.append(NFA({start, end}, {char}, transitions, start, {end}))

        elif char == '+':  # Union
            if len(stack) < 2:
                raise ValueError("Invalid Regular Expression: Union requires two operands.")
            nfa2, nfa1 = stack.pop(), stack.pop()
            start, end = new_state(), new_state()
            transitions = defaultdict(set)
            
            # Add epsilon transitions from new start to both NFAs
            transitions[(start, '')].add(nfa1.start_state)
            transitions[(start, '')].add(nfa2.start_state)
            
            # Add epsilon transitions from both NFA ends to new end
            for final_state in nfa1.final_states:
                transitions[(final_state, '')].add(end)
            for final_state in nfa2.final_states:
                transitions[(final_state, '')].add(end)
            
            # Add existing transitions
            for (state, symbol), next_states in nfa1.transitions.items():
                transitions[(state, symbol)].update(next_states)
            for (state, symbol), next_states in nfa2.transitions.items():
                transitions[(state, symbol)].update(next_states)
            
            # Convert defaultdict to dict
            transitions_dict = {k: v for k, v in transitions.items()}
            
            stack.append(NFA(
                nfa1.states | nfa2.states | {start, end}, 
                nfa1.alphabet | nfa2.alphabet | {''},
                transitions_dict, 
                start, 
                {end}
            ))

        elif char == '*':  # Kleene Star
            if not stack:
                raise ValueError("Invalid Regular Expression: * requires one operand.")
            nfa = stack.pop()
            start, end = new_state(), new_state()
            transitions = defaultdict(set)
            
            # Add epsilon transitions
            transitions[(start, '')].add(nfa.start_state)  # Start to NFA start
            transitions[(start, '')].add(end)  # Start to end (skip NFA)
            
            # NFA final states to NFA start (loop back)
            for final_state in nfa.final_states:
                transitions[(final_state, '')].add(nfa.start_state)
                transitions[(final_state, '')].add(end)  # NFA end to new end
            
            # Add existing transitions
            for (state, symbol), next_states in nfa.transitions.items():
                transitions[(state, symbol)].update(next_states)
            
            # Convert defaultdict to dict
            transitions_dict = {k: v for k, v in transitions.items()}
            
            stack.append(NFA(
                nfa.states | {start, end}, 
                nfa.alphabet | {''},
                transitions_dict, 
                start, 
                {end}
            ))

        elif char == '.':  # Concatenation
            if len(stack) < 2:
                raise ValueError("Invalid Regular Expression: Concatenation requires two operands.")
            nfa2, nfa1 = stack.pop(), stack.pop()
            transitions = defaultdict(set)
            
            # Add existing transitions
            for (state, symbol), next_states in nfa1.transitions.items():
                transitions[(state, symbol)].update(next_states)
            for (state, symbol), next_states in nfa2.transitions.items():
                transitions[(state, symbol)].update(next_states)
            
            # Connect nfa1's final states to nfa2's start state
            for final_state in nfa1.final_states:
                transitions[(final_state, '')].add(nfa2.start_state)
            
            # Convert defaultdict to dict
            transitions_dict = {k: v for k, v in transitions.items()}
            
            stack.append(NFA(
                nfa1.states | nfa2.states, 
                nfa1.alphabet | nfa2.alphabet | {''},
                transitions_dict, 
                nfa1.start_state, 
                nfa2.final_states
            ))

    if len(stack) != 1:
        raise ValueError("Invalid Regular Expression: Incorrect syntax.")
    
    return stack.pop()
def display_dfa(dfa):
    """Display the DFA details."""
    print("\nDFA Details:")
    print("States:", dfa["states"])
    print("Alphabet:", dfa["alphabet"])
    print("Start State:", dfa["start_state"])
    print("Final States:", dfa["final_states"])
    print("\nDFA Transition Table:")
    
    # Get all states and symbols for table header
    all_states = sorted(list(dfa["states"]))
    all_symbols = sorted(list(dfa["alphabet"]))
    
    # Print header
    header = f"{'State':<8} | " + " | ".join(f"{sym:<5}" for sym in all_symbols)
    print(header)
    print("-" * len(header))
    
    # Print transitions
    for state in all_states:
        row = f"{state:<8} | "
        for symbol in all_symbols:
            next_state = dfa["transitions"].get((state, symbol), "-")
            row += f"{next_state:<5} | "
        print(row[:-2])  # Remove trailing "| "

def visualize_automaton(automaton, title, is_nfa=True):
    """Create a graphical representation of the automaton using Graphviz."""
    dot = graphviz.Digraph(comment=title)
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add states
    states = automaton.states if is_nfa else automaton["states"]
    final_states = automaton.final_states if is_nfa else automaton["final_states"]
    start_state = automaton.start_state if is_nfa else automaton["start_state"]
    
    # Add all states
    for state in states:
        if state in final_states:
            dot.node(str(state), shape='doublecircle')
        else:
            dot.node(str(state), shape='circle')
    
    # Add a special start state pointer
    dot.node('start', shape='none', label='')
    dot.edge('start', str(start_state))
    
    # Add transitions
    if is_nfa:
        for (state, symbol), next_states in automaton.transitions.items():
            for next_state in next_states:
                label = 'ε' if symbol == '' else symbol
                dot.edge(str(state), str(next_state), label=label)
    else:
        for (state, symbol), next_state in automaton["transitions"].items():
            dot.edge(str(state), str(next_state), label=symbol)
    
    return dot

def minimize_dfa(dfa):
    """Minimize the DFA using Hopcroft's algorithm."""
    print("\n===== STEP 3: Minimize DFA =====")
    print("Minimizing DFA using Hopcroft's algorithm...")
    
    # Initialize the partition with final and non-final states
    final_states = dfa["final_states"]
    non_final_states = dfa["states"] - final_states
    partitions = []
    
    if final_states:
        partitions.append(final_states)
    if non_final_states:
        partitions.append(non_final_states)
    
    # Repeatedly refine the partition until no more refinements are possible
    changed = True
    while changed:
        changed = False
        new_partitions = []
        
        for partition in partitions:
            if len(partition) <= 1:
                new_partitions.append(partition)
                continue
                
            # Try to split the partition
            splits = {}
            reference_state = list(partition)[0]
            splits[reference_state] = {reference_state}
            
            for state in list(partition)[1:]:
                is_equivalent = True
                
                for symbol in dfa["alphabet"]:
                    # Get transitions for both states
                    ref_next = dfa["transitions"].get((reference_state, symbol))
                    state_next = dfa["transitions"].get((state, symbol))
                    
                    # If transitions go to different partitions, states are not equivalent
                    if ref_next != state_next:
                        is_equivalent = False
                        break
                
                if is_equivalent:
                    splits[reference_state].add(state)
                else:
                    # Create a new split
                    if state not in splits:
                        splits[state] = {state}
            
            # If we created more than one split, the partition has changed
            if len(splits) > 1:
                changed = True
                for split_states in splits.values():
                    new_partitions.append(split_states)
            else:
                new_partitions.append(partition)
        
        partitions = new_partitions
    
    # Create the minimized DFA
    state_mapping = {}
    for i, partition in enumerate(partitions):
        for state in partition:
            state_mapping[state] = f"q{i}"
    
    min_states = {state_mapping[state] for state in dfa["states"]}
    min_start_state = state_mapping[dfa["start_state"]]
    min_final_states = {state_mapping[state] for state in dfa["final_states"]}
    
    min_transitions = {}
    for (state, symbol), next_state in dfa["transitions"].items():
        min_transitions[(state_mapping[state], symbol)] = state_mapping[next_state]
    
    min_dfa = {
        "states": min_states,"alphabet": dfa["alphabet"],
        "transitions": min_transitions,
        "start_state": min_start_state,
        "final_states": min_final_states
    }
    
    return min_dfa

def explain_nfa_to_dfa_conversion(nfa, dfa):
    """Provide a detailed explanation of the NFA to DFA conversion process."""
    print("\n===== EXPLANATION: NFA to DFA Conversion =====")
    print("The conversion from NFA to DFA follows these steps:")
    print("1. We start with the epsilon-closure of the NFA's start state.")
    print("2. For each state in our DFA (which is a set of NFA states):")
    print("  a. We compute the transition for each symbol in the alphabet.")
    print("  b. For each NFA state in the set, we find all possible next states.")
    print("  c. We add the epsilon-closure of each of these next states.")
    print("3. We repeat until no new DFA states are discovered.")
    print("4. DFA states containing at least one NFA final state become final states in the DFA.")
    
    print("\nState Mappings from NFA to DFA:")
    # Since we don't have the actual mapping from the conversion, we'll infer it
    for dfa_state in dfa["states"]:
        print(f"  DFA State {dfa_state} represents a set of NFA states")
    
    print("\nDFA Acceptance:")
    print(f"  Final states in the DFA: {dfa['final_states']}")
    print("  The DFA accepts a string if processing the string ends in a final state.")

def explain_regex_language(regex, alphabet):
    """Explain the language represented by the regular expression."""
    print("\n===== EXPLANATION: Regular Expression Language =====")
    print(f"The regular expression '{regex}' represents a language over the alphabet {alphabet}.")
    print("The language consists of all strings that match the pattern described by the regex.")
    
    print("\nRegex Symbols:")
    print("  + : Union/Alternation (OR)")
    print("  * : Kleene Star (zero or more repetitions)")
    print("  . : Concatenation (implied in the regex)")
    print("  () : Grouping")
    
    print("\nExamples of strings that might be in the language:")
    # This is a simple attempt to generate some example strings
    # For a more accurate approach, you'd need to actually trace through the DFA
    print("  (This is a simplified guess based on the regex pattern)")
    if "*" in regex:
        print("  - Since the regex contains '*', it may accept empty strings or repetitions.")
    if "+" in regex:
        print("  - Since the regex contains '+', it accepts alternatives.")

def test_dfa_with_string(dfa, test_string):
    """Test the DFA with a given input string and return the result."""
    current_state = dfa["start_state"]
    accepted = True
    trace = [f"Start state: {current_state}"]

    for symbol in test_string:
        if symbol not in dfa["alphabet"]:
            return f"Error: Symbol '{symbol}' is not in the alphabet {dfa['alphabet']}", False, trace
        if (current_state, symbol) not in dfa["transitions"]:
            return f"Error: No transition defined for state {current_state} with symbol '{symbol}'", False, trace
        next_state = dfa["transitions"][(current_state, symbol)]
        trace.append(f"  Input '{symbol}': {current_state} -> {next_state}")
        current_state = next_state

    if accepted:
        if current_state in dfa["final_states"]:
            return f"String '{test_string}' is ACCEPTED by the DFA.\nEnded in accepting state: {current_state}", True, trace
        else:
            return f"String '{test_string}' is REJECTED by the DFA.\nEnded in non-accepting state: {current_state}", False, trace

def display_dfa_table(dfa):
    """Display the DFA's transition table with start and final state markers."""
    transitions = dfa["transitions"]
    alphabet = sorted(list(filter(lambda x: x != '', dfa["alphabet"])))
    states = sorted(list(set([k[0] for k in transitions.keys()] + [v for v in transitions.values()])))
    table_data = []
    header = ["State"] + alphabet
    table_data.append(header)

    for state in states:
        row = [state]
        if state == dfa["start_state"]:
            row[0] += " (Start)"
        if state in dfa["final_states"]:
            row[0] += " (Final)"
        for symbol in alphabet:
            if (state, symbol) in transitions:
                row.append(transitions[(state, symbol)])
            else:
                row.append("-")  # Indicate no transition
        table_data.append(row)

    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    st.table(df)

def capture_display(display_func):
    """Capture the output of the display functions."""
    import io
    import sys
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    display_func()
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    return output

def explain_regex_language(regex, alphabet):
    """Provide an informative and engaging explanation of the regular expression language."""
    explanation = f"""
    ## Unraveling the Language of Regular Expression: '{regex}'

    Imagine you're trying to describe a set of strings with a concise pattern. That's exactly what regular expressions do!

    **The Alphabet:**
    Our language uses the alphabet: {alphabet}

    **What Does It Mean?**
    The regular expression '{regex}' defines a language consisting of strings that match its pattern. Think of it as a set of rules that strings must follow to be accepted.

    **Key Regex Symbols:**
    - `|` (Union/Alternation): Represents "or". Example: `a|b` matches either 'a' or 'b'.
    - `*` (Kleene Star): Represents zero or more repetitions. Example: `a*` matches '', 'a', 'aa', 'aaa', etc.
    - `.` (Concatenation): Implied between symbols. Example: `ab` matches 'ab'.
    - `()` (Grouping): Groups parts of the regex. Example: `(ab)*` matches '', 'ab', 'abab', etc.

    **Curiosity Corner:**
    Did you know that regular expressions are used in text editors, programming languages, and network security? They're a powerful tool for pattern matching!
    """
    return explanation

def explain_nfa_to_dfa_conversion(nfa, dfa):
    """Provide an informative and engaging explanation of the NFA to DFA conversion."""
    explanation = f"""
    ## The Magic of NFA to DFA Conversion

    Have you ever wondered how a computer understands complex patterns described by regular expressions? The answer lies in the conversion from Non-deterministic Finite Automata (NFA) to Deterministic Finite Automata (DFA).

    **Why Convert?**
    - NFAs can be non-deterministic, meaning they can have multiple possible next states for a given input.
    - DFAs are deterministic, making them easier for computers to process.

    **The Conversion Process:**
    1.  **Epsilon Closure:** We start with the epsilon-closure of the NFA's start state.
    2.  **Subset Construction:** For each state in our DFA (which is a set of NFA states):
        - Compute the transition for each symbol in the alphabet.
        - For each NFA state in the set, find all possible next states.
        - Add the epsilon-closure of each of these next states.
    3.  **Repeat:** We repeat until no new DFA states are discovered.
    4.  **Final States:** DFA states containing at least one NFA final state become final states in the DFA.

    **DFA Acceptance:**
    The DFA accepts a string if processing the string ends in a final state.

    **Curiosity Corner:**
    The NFA to DFA conversion is a fundamental concept in computer science and is used in various applications, including compilers and network protocols.
    """
    return explanation

def main():
    st.set_page_config(page_title="Regex to Minimized DFA", layout="wide")

    st.title("Regular Expression to Minimized DFA Converter")

    regex = st.text_input("Enter Regular Expression (e.g., (a+b)*a(a+b)):")

    # Initial Sidebar Content
    with st.sidebar:
        st.header("Project Introduction")
        st.markdown("""
        Welcome to the Regular Expression to Minimized DFA Converter! This application allows you to:
        - Convert regular expressions to Non-deterministic Finite Automata (NFA).
        - Convert NFAs to Deterministic Finite Automata (DFA).
        - Minimize DFAs to their simplest form.
        - Understand the language defined by a regular expression.
        - Test strings against the generated DFAs.
        """)

    if regex:
        try:
            st.write("## Processing Regular Expression:", regex)

            # Convert RE to NFA-ε
            st.subheader("Step 1: Convert Regular Expression to NFA-ε")
            nfa_epsilon = re_to_nfa(regex)

            if nfa_epsilon:
                st.markdown("### NFA-ε Details:")
                nfa_epsilon_str = capture_display(nfa_epsilon.display)
                st.code(nfa_epsilon_str, language="text")

                st.markdown("### NFA-ε Visualization:")
                nfa_viz = visualize_automaton(nfa_epsilon, f"NFA for {regex}", is_nfa=True)
                nfa_viz.render("nfa", format="png", cleanup=True)
                st.image("nfa.png", use_column_width=True)

                # Convert NFA-ε to DFA
                st.subheader("Step 2: Convert NFA-ε to DFA")
                dfa = nfa_epsilon.to_dfa()

                st.markdown("### Unminimized DFA Details:")
                st.write("#### Transition Table:")
                display_dfa_table(dfa)

                st.markdown("### Unminimized DFA Visualization:")
                dfa_viz = visualize_automaton(dfa, f"DFA for {regex}", is_nfa=False)
                dfa_viz.render("dfa", format="png", cleanup=True)
                st.image("dfa.png", use_column_width=True)

                # Minimize DFA
                st.subheader("Step 3: Minimize DFA")
                min_dfa = minimize_dfa(dfa)

                st.markdown("### Minimized DFA Details:")
                st.write("#### Transition Table:")
                display_dfa_table(min_dfa)

                st.markdown("### Minimized DFA Visualization:")
                min_dfa_viz = visualize_automaton(min_dfa, f"Minimized DFA for {regex}", is_nfa=False)
                min_dfa_viz.render("min_dfa", format="png", cleanup=True)
                st.image("min_dfa.png", use_column_width=True)


                # Explanations in Sidebar
                with st.sidebar:
                    st.header("Explanations")

                    with st.expander("NFA to DFA Conversion"):
                        explanation_str = explain_nfa_to_dfa_conversion(nfa_epsilon, dfa)
                        st.markdown(explanation_str)

                    with st.expander("Regular Expression Language"):
                        alphabet = set(filter(lambda x: x != '', nfa_epsilon.alphabet))
                        regex_explanation_str = explain_regex_language(regex, alphabet)
                        st.markdown(regex_explanation_str)

                st.subheader("Test DFA with String")
                test_string = st.text_input("Enter a string to test:")
                if test_string:
                    result, accepted, trace = test_dfa_with_string(min_dfa, test_string)
                    st.write(result)
                    st.markdown("### Trace:")
                    st.code("\n".join(trace), language="text")

            else:
                st.error("Error: Could not create NFA from the regular expression.")

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
	st.markdown("""
	---
	**About Me:** This application was created by Pruthak Jani.
	
	**Connect with Me:** If you have any questions, feedback, bug reports, or if you find any incorrect answers, please feel free to reach out via email: pruthak.jani@gmail.com
	""")


if __name__ == "__main__":
    main()
