import streamlit as st
import itertools
from collections import defaultdict
import graphviz
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
        # Also handle '^' as epsilon character
        if (
            (regex[i].isalnum() or regex[i] == ')' or regex[i] == '*' or regex[i] == '^') and
            (regex[i+1].isalnum() or regex[i+1] == '(' or regex[i+1] == '^')
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
        if char.isalnum() or char == '^':  # Include ^ as a character, not an operator
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
        if char.isalnum() or char == '^':  # Single character automaton or epsilon
            start, end = new_state(), new_state()
            symbol = '' if char == '^' else char  # Use empty string for epsilon
            transitions = {(start, symbol): {end}}
            stack.append(NFA({start, end}, {symbol}, transitions, start, {end}))

        elif char == '+':  # Union
            if len(stack) < 2:
                # Handle special case for union with epsilon (like a+^)
                if len(stack) == 1:
                    nfa1 = stack.pop()
                    # Create an epsilon NFA
                    start_eps, end_eps = new_state(), new_state()
                    epsilon_transitions = {(start_eps, ''): {end_eps}}
                    nfa2 = NFA({start_eps, end_eps}, {''}, epsilon_transitions, start_eps, {end_eps})
                    
                    # Now continue with the normal union processing
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
                    continue
                else:
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

def explain_nfa_to_dfa_conversion(nfa_obj, dfa):
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
    trace = [f"Start state: {current_state}"]

    # Handle empty string test case
    if not test_string:
        is_accepted = current_state in dfa["final_states"]
        result = f"Empty string is {'ACCEPTED' if is_accepted else 'REJECTED'} by the DFA."
        result += f"\nEnded in {'accepting' if is_accepted else 'non-accepting'} state: {current_state}"
        return result, is_accepted, trace

    for symbol in test_string:
        if symbol not in dfa["alphabet"]:
            return f"Error: Symbol '{symbol}' is not in the alphabet {dfa['alphabet']}", False, trace
        if (current_state, symbol) not in dfa["transitions"]:
            return f"Error: No transition defined for state {current_state} with symbol '{symbol}'", False, trace
        next_state = dfa["transitions"][(current_state, symbol)]
        trace.append(f"  Input '{symbol}': {current_state} -> {next_state}")
        current_state = next_state

    is_accepted = current_state in dfa["final_states"]
    if is_accepted:
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
    nfa_states_count = len(nfa.states)
    dfa_states_count = len(dfa["states"])
    efficiency_gain = f"{100 - (dfa_states_count / (2**nfa_states_count) * 100):.1f}%" if nfa_states_count > 0 else "N/A"
    
    explanation = f"""
    ## The Magic of NFA to DFA Conversion

    Have you ever wondered how a computer understands complex patterns described by regular expressions? The answer lies in the conversion from Non-deterministic Finite Automata (NFA) to Deterministic Finite Automata (DFA).

    **NFA Characteristics:**
    - States: {nfa_states_count} states
    - Alphabet: {sorted(list(filter(lambda x: x != '', nfa.alphabet)))}
    - Start State: {nfa.start_state}
    - Final States: {nfa.final_states}
    - Non-deterministic: Can have multiple possible next states for a given input
    - Includes ε-transitions: {'' in nfa.alphabet}

    **DFA Characteristics:**
    - States: {dfa_states_count} states (theoretically could have up to 2^{nfa_states_count} states)
    - Alphabet: {sorted(list(dfa["alphabet"]))}
    - Start State: {dfa["start_state"]}
    - Final States: {dfa["final_states"]}
    - Deterministic: Exactly one next state for each input
    - No ε-transitions

    **The Conversion Process:**
    1.  **Epsilon Closure:** We started with the epsilon-closure of the NFA's start state '{nfa.start_state}'.
    2.  **Subset Construction:** For each state in our DFA (which is a set of NFA states):
        - Computed the transition for each symbol in the alphabet.
        - For each NFA state in the set, found all possible next states.
        - Added the epsilon-closure of each of these next states.
    3.  **Repeat:** We repeated until no new DFA states were discovered.
    4.  **Final States:** DFA states containing at least one NFA final state became final states in the DFA.

    **DFA Acceptance:**
    The DFA accepts a string if processing the string ends in one of these final states: {dfa["final_states"]}

    **Efficiency Analysis:**
    - Maximum possible DFA states: 2^{nfa_states_count} = {2**nfa_states_count}
    - Actual DFA states: {dfa_states_count}
    - State reduction: {efficiency_gain} compared to worst case

    **Curiosity Corner:**
    The NFA to DFA conversion is a fundamental concept in computer science and is used in various applications, including compilers and network protocols.
    """
    return explanation

def main():
    st.set_page_config(
        page_title="Automata Architect | RE to DFA Converter", 
        page_icon="🧠",
        layout="wide"
    )

    st.title("Automata Architect: Regular Expression to Minimal DFA Converter")
    st.markdown("""
    This comprehensive tool transforms regular expressions into their canonical finite automata representations through 
    a systematic pipeline of conversions: Regular Expression → NFA-ε → DFA → Minimized DFA.
    
    Designed for computer science students, educators, and professionals, this interactive system visualizes each transformation
    step with detailed explanations of the underlying theoretical concepts in formal language theory and automata theory.
    """)

    regex = st.text_input("Enter Regular Expression:", 
                        placeholder="e.g., (a+b)*a(a+b)",
                         help="Use + for union, * for Kleene star, ^ for epsilon, and parentheses for grouping")
    
    custom_alphabet = st.text_input("Custom Alphabet (optional, comma-separated):", 
                                    placeholder="e.g., a,b,c",
                                    help="Leave blank to automatically detect from regex. All symbols used in regex must be in this alphabet.")
    if not regex:
        with st.container():
                    st.markdown("""
                    👋 Hi, I'm **Pruthak Jani** — an aspiring **AI/ML Engineer & Data Scientist**.  
                    I created this **Automata Architect** to help students and professionals **understand Automata Theory**  
                    by providing **step-by-step DFA/NFA conversions, minimization techniques, and learning insights.**  

                    **Connect:** For questions, feedback, bug reports, or academic discussions: pruthak.jani@gmail.com

                    If you found this useful, feel free to **connect with me** on LinkedIn!  
                    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Pruthak%20Jani-blue?logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)

                    **Happy Learning & Keep Innovating! 🚀**
                    """, unsafe_allow_html=True)

    # Enhanced Sidebar Content
    with st.sidebar:
        st.header("📚 Automata Theory Guide")
        st.markdown("""
        ### What is this tool?
        This application demonstrates the fundamental processes in compiler design and formal language theory:
        
        1. **Regular Expression Parsing**: Converting human-readable patterns to formal representations
        2. **Thompson's Construction**: Building NFAs from regular expressions
        3. **Subset Construction**: Converting NFAs to DFAs
        4. **Hopcroft's Algorithm**: Minimizing DFAs to their canonical form
        
        ### Special Notation:
        - `+` for union/alternation (OR)
        - `*` for Kleene star (zero or more repetitions)
        - `^` for epsilon/empty string (ε)
        - Parentheses `()` for grouping expressions
        
        ### Academic Applications:
        - Lexical analysis in compiler design
        - Pattern matching in text processing
        - Protocol validation in communication systems
        - Natural language processing fundamentals
        """)

    if regex:
        try:
            # Process custom alphabet if provided
            alphabet_set = set()
            if custom_alphabet:
                alphabet_list = [a.strip() for a in custom_alphabet.split(',')]
                alphabet_set = set(alphabet_list)
                if not alphabet_set:
                    st.warning("Custom alphabet is empty. Using automatic alphabet detection.")
            
            # Validate that all regex symbols are in the custom alphabet if provided
            if alphabet_set:
                # Extract all alphanumeric characters from regex
                regex_symbols = set()
                for char in regex:
                    if char.isalnum() and char not in ('+', '*'):  # Exclude operators
                        regex_symbols.add(char)
                
                # Check if all regex symbols are in the custom alphabet
                undefined_symbols = regex_symbols - alphabet_set
                if undefined_symbols:
                    st.error(f"Error: The following symbols in your regex are not defined in your custom alphabet: {', '.join(sorted(undefined_symbols))}. Please add them to your alphabet or update your regex.")
                    return
            
            # Replace visible ^ with ε for display, but keep ^ in the processing
            display_regex = regex.replace('^', 'ε')
            st.write("## Processing Regular Expression:", display_regex)
            
            if alphabet_set:
                st.write("## Using Custom Alphabet:", ", ".join(sorted(list(alphabet_set))))
            
            # Convert RE to NFA-ε
            st.subheader("Step 1: Convert Regular Expression to NFA-ε")
            st.markdown("""
            The first step uses **Thompson's Construction** to convert the regular expression into a 
            Non-deterministic Finite Automaton with epsilon transitions (NFA-ε). This construction guarantees 
            that the resulting automaton precisely recognizes the language defined by the regex.
            """)
            
            nfa_epsilon = re_to_nfa(regex)
            
            # Update NFA alphabet with custom alphabet if provided
            if nfa_epsilon and alphabet_set:
                # Add the custom alphabet while preserving epsilon transitions
                epsilon_exists = '' in nfa_epsilon.alphabet
                nfa_epsilon.alphabet = alphabet_set
                if epsilon_exists:
                    nfa_epsilon.alphabet.add('')
            
            if nfa_epsilon:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### NFA-ε Details:")
                    nfa_epsilon_str = capture_display(nfa_epsilon.display)
                    st.code(nfa_epsilon_str, language="text")
                
                with col2:
                    st.markdown("### NFA-ε Visualization:")
                    nfa_viz = visualize_automaton(nfa_epsilon, f"NFA for {regex}", is_nfa=True)
                    nfa_viz.render("nfa", format="png", cleanup=True)
                    st.image("nfa.png", use_container_width=True)
                    st.caption("Each circle represents a state. Double circles are accepting states. Arrows show transitions between states.")

                # Show NFA-ε to NFA Conversion Steps
                st.subheader("Step 1.5: NFA-ε to NFA Conversion (Eliminating ε-transitions)")
                st.markdown("""
                Before converting to a DFA, we can first eliminate epsilon transitions from the NFA.
                This intermediate step helps understand the overall conversion process better.
                """)
                
                with st.expander("Show NFA-ε to NFA Conversion Steps"):
                    # Compute epsilon closures for all states
                    epsilon_closures = {}
                    for state in nfa_epsilon.states:
                        epsilon_closures[state] = nfa_epsilon.epsilon_closure(state)
                    
                    # Display epsilon closures in a table
                    st.markdown("### Step 1: Compute ε-closures for all states")
                    closure_data = {"State": [], "ε-closure": []}
                    for state, closure in epsilon_closures.items():
                        closure_data["State"].append(state)
                        closure_data["ε-closure"].append(', '.join(sorted([str(s) for s in closure])))
                    
                    st.table(pd.DataFrame(closure_data))
                    st.markdown("""
                    **How to compute ε-closure:**
                    - Start with the state itself
                    - Add all states reachable by following ε-transitions
                    - Continue until no new states can be added
                    """)
                    
                    # Show how to create the new transitions
                    st.markdown("### Step 2: Create new transition table for NFA without ε-transitions")
                    st.markdown("""
                    For each state S and input symbol a, the new transition δ'(S,a) will be:
                    The union of δ(R,a) for all states R in the ε-closure of S
                    
                    Follow these steps for each state and symbol:
                    1. Find the ε-closure of the state
                    2. For each state in the closure, find all states reachable with the symbol
                    3. For each of those states, include their ε-closures in the result
                    """)
                    
                    # Create a mock transition table for the NFA without ε transitions
                    alphabet_without_epsilon = nfa_epsilon.alphabet - {''}
                    nfa_transitions_data = {"State": []}
                    for symbol in alphabet_without_epsilon:
                        nfa_transitions_data[symbol] = []
                    
                    for state in sorted(list(nfa_epsilon.states)):
                        nfa_transitions_data["State"].append(state)
                        for symbol in alphabet_without_epsilon:
                            new_states = set()
                            for s in epsilon_closures[state]:
                                if (s, symbol) in nfa_epsilon.transitions:
                                    for target in nfa_epsilon.transitions[(s, symbol)]:
                                        new_states.update(epsilon_closures[target])
                            
                            if new_states:
                                nfa_transitions_data[symbol].append(', '.join(sorted([str(s) for s in new_states])))
                            else:
                                nfa_transitions_data[symbol].append("∅")  # Empty set
                    
                    st.table(pd.DataFrame(nfa_transitions_data))
                    
                    # Show final states calculation
                    st.markdown("### Step 3: Determine the final states of the new NFA")
                    st.markdown("""
                    A state in the new NFA is a final state if its ε-closure contains any final state 
                    from the original NFA-ε.
                    """)
                    
                    final_states_data = {"State": [], "Contains Final State?": [], "Is Final in New NFA": []}
                    for state in sorted(list(nfa_epsilon.states)):
                        final_states_data["State"].append(state)
                        
                        contains_final = any(s in nfa_epsilon.final_states for s in epsilon_closures[state])
                        final_states_data["Contains Final State?"].append("Yes" if contains_final else "No")
                        final_states_data["Is Final in New NFA"].append("Yes" if contains_final else "No")
                    
                    st.table(pd.DataFrame(final_states_data))

                # Convert NFA-ε to DFA
                st.subheader("Step 2: Convert NFA-ε to DFA")
                st.markdown("""
                Using the **Subset Construction Algorithm**, we convert the NFA-ε to a Deterministic Finite Automaton (DFA).
                This eliminates non-determinism by creating states in the DFA that correspond to sets of NFA states.
                Each DFA state represents all possible states the NFA could be in after reading a particular input.
                """)
                
                dfa = nfa_epsilon.to_dfa()

                with st.expander("Show NFA to DFA Subset Construction Steps"):
                    st.markdown("### Step 1: Start with the ε-closure of the NFA start state")
                    initial_closure = nfa_epsilon.epsilon_closure(nfa_epsilon.start_state)
                    initial_closure_str = ', '.join(sorted([str(s) for s in initial_closure]))
                    st.markdown(f"**Initial DFA state:** {{{initial_closure_str}}} (Label this as q0)")
                    
                    st.markdown("### Step 2: Apply subset construction algorithm")
                    st.markdown("""
                    1. Create a queue of unprocessed DFA states (each being a set of NFA states)
                    2. For each unprocessed DFA state and each input symbol:
                        - Find all states the NFA can transition to
                        - Include their ε-closures
                        - This forms a new DFA state if not seen before
                    3. Continue until all DFA states are processed
                    """)
                    
                    # Recreate the subset construction process step by step
                    alphabet_without_epsilon = nfa_epsilon.alphabet - {''}
                    
                    # Prepare data structures
                    subset_table = {
                        "Step": [],
                        "Current Subset": [],
                        "Input": [],
                        "Next States Before ε-closure": [],
                        "Next States After ε-closure": [],
                        "New DFA State?": []
                    }
                    
                    # Mock subset construction steps (simplified for demonstration)
                    initial_closure_frozen = frozenset(initial_closure)
                    unprocessed = [initial_closure_frozen]
                    processed = set()
                    dfa_state_mapping = {initial_closure_frozen: "q0"}
                    step_counter = 1
                    
                    while unprocessed:
                        current = unprocessed.pop(0)
                        if current in processed:
                            continue
                        
                        processed.add(current)
                        current_label = dfa_state_mapping[current]
                        
                        for symbol in sorted(alphabet_without_epsilon):
                            # Find next states before ε-closure
                            next_states_before = set()
                            for nfa_state in current:
                                if (nfa_state, symbol) in nfa_epsilon.transitions:
                                    next_states_before.update(nfa_epsilon.transitions[(nfa_state, symbol)])
                            
                            # Apply ε-closure to get the complete next state
                            next_states_after = set()
                            for ns in next_states_before:
                                next_states_after.update(nfa_epsilon.epsilon_closure(ns))
                            
                            next_states_frozen = frozenset(next_states_after)
                            
                            # Determine if this is a new DFA state
                            is_new = next_states_frozen and next_states_frozen not in dfa_state_mapping
                            
                            if is_new:
                                dfa_state_mapping[next_states_frozen] = f"q{len(dfa_state_mapping)}"
                                unprocessed.append(next_states_frozen)
                            
                            # Record this step
                            subset_table["Step"].append(step_counter)
                            subset_table["Current Subset"].append(f"{current_label}: {{{', '.join(sorted([str(s) for s in current]))}}}")
                            subset_table["Input"].append(symbol)
                            
                            if next_states_before:
                                subset_table["Next States Before ε-closure"].append(f"{{{', '.join(sorted([str(s) for s in next_states_before]))}}}")
                            else:
                                subset_table["Next States Before ε-closure"].append("∅")
                                
                            if next_states_after:
                                subset_table["Next States After ε-closure"].append(f"{{{', '.join(sorted([str(s) for s in next_states_after]))}}}")
                                next_label = dfa_state_mapping[next_states_frozen]
                            else:
                                subset_table["Next States After ε-closure"].append("∅")
                                next_label = "dead state"
                            
                            subset_table["New DFA State?"].append(f"{'Yes: ' + next_label if is_new else 'No: ' + next_label}")
                            
                            step_counter += 1
                    
                    # Display the subset construction process table
                    st.table(pd.DataFrame(subset_table))
                    
                    st.markdown("### Step 3: Determine the DFA's final states")
                    st.markdown("""
                    A DFA state is a final state if any of the NFA states in its subset is a final state.
                    """)
                    
                    final_states_dfa = {
                        "DFA State": [],
                        "NFA States": [],
                        "Contains NFA Final State?": [],
                        "Is DFA Final State?": []
                    }
                    
                    for subset, label in dfa_state_mapping.items():
                        final_states_dfa["DFA State"].append(label)
                        final_states_dfa["NFA States"].append(f"{{{', '.join(sorted([str(s) for s in subset]))}}}")
                        
                        contains_final = any(s in nfa_epsilon.final_states for s in subset)
                        final_states_dfa["Contains NFA Final State?"].append("Yes" if contains_final else "No")
                        final_states_dfa["Is DFA Final State?"].append("Yes" if contains_final else "No")
                    
                    st.table(pd.DataFrame(final_states_dfa))

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### Unminimized DFA Details:")
                    st.write("#### Transition Table:")
                    display_dfa_table(dfa)
                    st.caption("This table shows all transitions in the DFA. Each row represents a state, and each column represents an input symbol.")
                
                with col2:
                    st.markdown("### Unminimized DFA Visualization:")
                    dfa_viz = visualize_automaton(dfa, f"DFA for {regex}", is_nfa=False)
                    dfa_viz.render("dfa", format="png", cleanup=True)
                    st.image("dfa.png", use_container_width=True)
                    st.caption("The DFA is deterministic - exactly one transition for each input symbol from any state.")

                # Minimize DFA
                st.subheader("Step 3: Minimize DFA")
                st.markdown("""
                The final step uses **Hopcroft's Algorithm** to minimize the DFA by combining equivalent states.
                Two states are equivalent if they have the same behavior for all possible input sequences.
                This results in the smallest possible DFA that still recognizes the same language.
                """)
                
                min_dfa = minimize_dfa(dfa)

                with st.expander("Show DFA Minimization Steps (Partition Method)"):
                    st.markdown("### Step 1: Initial Partition")
                    st.markdown("""
                    Start by dividing states into two groups:
                    - Final (accepting) states
                    - Non-final (non-accepting) states
                    
                    These form our initial partition P₀.
                    """)
                    
                    final_states = dfa["final_states"]
                    non_final_states = dfa["states"] - final_states
                    
                    initial_partition = {
                        "Group": ["Group 1 (Final)", "Group 2 (Non-Final)"],
                        "States": [
                            ', '.join(sorted(list(final_states))) if final_states else "∅",
                            ', '.join(sorted(list(non_final_states))) if non_final_states else "∅"
                        ]
                    }
                    
                    st.table(pd.DataFrame(initial_partition))
                    
                    st.markdown("### Step 2: Refine Partitions Iteratively")
                    st.markdown("""
                    For each iteration:
                    1. For each group in the current partition
                    2. Try to split the group based on transitions
                    3. Two states p and q remain in the same group if:
                        - For each input symbol a, δ(p,a) and δ(q,a) go to states in the same group
                    4. Continue until no more refinements are possible
                    """)
                    
                    # Create a mock partition refinement process
                    partition_steps = {
                        "Iteration": [],
                        "Current Partition": [],
                        "Refinement Process": [],
                        "New Partition": []
                    }
                    
                    # Initialize with our starting partition
                    partitions = []
                    if final_states:
                        partitions.append(final_states)
                    if non_final_states:
                        partitions.append(non_final_states)
                    
                    alphabet = dfa["alphabet"]
                    iteration = 1
                    
                    # Track partitions across iterations
                    all_partitions = [partitions.copy()]
                    
                    # Mock refinement for demonstration (simplified)
                    changed = True
                    while changed and iteration <= 5:  # Limit iterations to prevent infinite loops
                        changed = False
                        new_partitions = []
                        
                        partition_steps["Iteration"].append(iteration)
                        partition_steps["Current Partition"].append(
                            " | ".join(["{" + ", ".join(sorted(list(p))) + "}" for p in partitions])
                        )
                        
                        refinement_details = []
                        
                        for partition in partitions:
                            if len(partition) <= 1:
                                new_partitions.append(partition)
                                refinement_details.append(f"Group {{{', '.join(sorted(list(partition)))}}} has only one state - no refinement needed")
                                continue
                            
                            # Try to split the partition
                            splits = {}
                            states_list = sorted(list(partition))
                            
                            for i, state in enumerate(states_list):
                                found_group = False
                                
                                for group_representative, group in splits.items():
                                    equivalent = True
                                    
                                    # Check if state has same transitions as group_representative
                                    for symbol in alphabet:
                                        target1 = dfa["transitions"].get((state, symbol))
                                        target2 = dfa["transitions"].get((group_representative, symbol))
                                        
                                        # Find which partition each target belongs to
                                        partition1 = next((i for i, p in enumerate(partitions) if target1 in p), -1)
                                        partition2 = next((i for i, p in enumerate(partitions) if target2 in p), -1)
                                        
                                        if partition1 != partition2:
                                            equivalent = False
                                            break
                                    
                                    if equivalent:
                                        group.add(state)
                                        found_group = True
                                        break
                                
                                if not found_group:
                                    splits[state] = {state}
                            
                            # Add details about this refinement
                            if len(splits) > 1:
                                refinement_details.append(
                                    f"Group {{{', '.join(sorted(list(partition)))}}} splits into " + 
                                    " | ".join(["{" + ", ".join(sorted(list(g))) + "}" for g in splits.values()])
                                )
                                changed = True
                            else:
                                refinement_details.append(f"Group {{{', '.join(sorted(list(partition)))}}} cannot be refined further")
                            
                            # Add the splits to new partitions
                            for group in splits.values():
                                new_partitions.append(group)
                        
                        partition_steps["Refinement Process"].append("\n".join(refinement_details))
                        partition_steps["New Partition"].append(
                            " | ".join(["{" + ", ".join(sorted(list(p))) + "}" for p in new_partitions])
                        )
                        
                        partitions = new_partitions
                        all_partitions.append(partitions.copy())
                        iteration += 1
                        
                        if not changed:
                            break
                    
                    st.table(pd.DataFrame(partition_steps))
                    
                    st.markdown("### Step 3: Create the Minimized DFA")
                    st.markdown("""
                    1. Each group in the final partition becomes a state in the minimized DFA
                    2. Choose a representative from each group
                    3. For transitions, use the representative's transitions
                    4. The start state is the group containing the original start state
                    5. Final states are groups containing any original final state
                    """)
                    
                    # Show state mapping
                    final_mapping = {
                        "Group": [],
                        "States in Group": [],
                        "Minimized DFA State": []
                    }
                    
                    for i, group in enumerate(partitions):
                        final_mapping["Group"].append(f"Group {i+1}")
                        final_mapping["States in Group"].append(', '.join(sorted(list(group))))
                        final_mapping["Minimized DFA State"].append(f"q{i}")
                    
                    st.table(pd.DataFrame(final_mapping))

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### Minimized DFA Details:")
                    st.write("#### Transition Table:")
                    display_dfa_table(min_dfa)
                    
                    # Add state reduction statistics
                    original_states = len(dfa["states"])
                    minimized_states = len(min_dfa["states"])
                    if original_states > minimized_states:
                        reduction = ((original_states - minimized_states) / original_states) * 100
                        st.success(f"State reduction: {reduction:.1f}% (from {original_states} to {minimized_states} states)")
                    else:
                        st.info("The DFA was already minimal - no state reduction possible.")
                
                with col2:
                    st.markdown("### Minimized DFA Visualization:")
                    min_dfa_viz = visualize_automaton(min_dfa, f"Minimized DFA for {regex}", is_nfa=False)
                    min_dfa_viz.render("min_dfa", format="png", cleanup=True)
                    st.image("min_dfa.png", use_container_width=True)
                    st.caption("This is the canonical (minimal) DFA for the given regex - no smaller DFA can recognize the same language.")

                # Explanations in Sidebar
                with st.sidebar:
                    st.header("💡 Theoretical Insights")

                    with st.expander("NFA to DFA Conversion"):
                        explanation_str = explain_nfa_to_dfa_conversion(nfa_epsilon, dfa)
                        st.markdown(explanation_str)
                    
                    with st.expander("Regular Expression Theory"):
                        st.markdown("""
                        Regular expressions define Type-3 languages in the Chomsky hierarchy, the simplest 
                        class of formal languages. These languages can be recognized by finite automata,
                        making them computationally efficient to process.
                        
                        The power of regular expressions comes with limitations - they cannot recognize:
                        - Balanced parentheses (like programming languages)
                        - Palindromes of arbitrary length
                        - Languages requiring counting beyond a fixed bound
                        
                        For these more complex patterns, context-free grammars and pushdown automata are required.
                        """)

                # Test DFA with String
                st.subheader("Test DFA with String")
                st.markdown("""
                Enter any string to test if it's accepted by the minimized DFA. The trace shows the step-by-step
                state transitions as each character is processed.
                """)
                
                test_string = st.text_input("Enter a string to test:", 
                                        help="Use only symbols from the alphabet. Empty string is represented by not entering anything.")
                
                if test_string is not None:  # Allow testing empty string
                    # Validate test string against alphabet
                    invalid_chars = set()
                    for char in test_string:
                        if char not in min_dfa["alphabet"]:
                            invalid_chars.add(char)
                    
                    if invalid_chars:
                        st.error(f"Error: Test string contains symbols not in the alphabet: {', '.join(invalid_chars)}")
                    else:
                        result, accepted, trace = test_dfa_with_string(min_dfa, test_string)
                        
                        if accepted:
                            st.success(result)
                        else:
                            st.error(result)
                            
                        with st.expander("Show Processing Trace"):
                            st.code("\n".join(trace), language="text")
                            st.caption("This trace shows the step-by-step execution of the DFA on the input string.")

            else:
                st.error("Error: Could not create NFA from the regular expression. Please check your syntax.")

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.exception(e)  # Show detailed exception in development
        
        # Enhanced footer with academic references
        st.markdown("""
        ---
        ### References and Further Reading
        - Hopcroft, J.E., Motwani, R., & Ullman, J.D. (2006). *Introduction to Automata Theory, Languages, and Computation* (3rd ed.). Pearson.
        - Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.
        - Martin, J.C. (2010). *Introduction to Languages and the Theory of Computation* (4th ed.). McGraw-Hill. 
        """)
        # Add Footer
        st.markdown("""
        ---
        👋 Hi, I'm **Pruthak Jani** — an aspiring **AI/ML Engineer & Data Scientist**.  
        I created this **Automata Architect** to help students and professionals **understand Automata Theory**  
        by providing **step-by-step DFA/NFA conversions, minimization techniques, and learning insights.**  

        **Connect:** For questions, feedback, bug reports, or academic discussions: pruthak.jani@gmail.com

        If you found this useful, feel free to **connect with me** on LinkedIn!  
        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Pruthak%20Jani-blue?logo=linkedin)](https://www.linkedin.com/in/pruthak-jani/)

        **Happy Learning & Keep Innovating! 🚀**
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
