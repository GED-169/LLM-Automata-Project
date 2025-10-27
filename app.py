# app.py
# Import necessary libraries
import streamlit as st
import re
import pandas as pd
from graphviz import Digraph

#FINITE AUTOMATA INTERPRETER CLASSES
class Lexer:
    """Breaks the source code into a stream of tokens."""
    def __init__(self, source_code):
        self.source_code = source_code; self.tokens = []; self.current_line = 1; self.current_col = 1
    def tokenize(self):
        token_specs = [('COMMENT',r'#.*'), ('KEYWORD',r'\b(states|alphabet|start|final|transitions|end|eps)\b'), ('IDENTIFIER',r'[a-zA-Z0-9_]+'), ('ARROW',r'->'), ('LPAREN',r'\('), ('RPAREN',r'\)'), ('COMMA',r','), ('COLON',r':'), ('NEWLINE',r'\n'), ('WHITESPACE',r'[ \t]+'), ('UNKNOWN',r'.')]
        token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specs)
        for match in re.finditer(token_regex, self.source_code):
            token_type = match.lastgroup; token_value = match.group()
            if token_type == 'NEWLINE': self.current_line += 1; self.current_col = 1
            else:
                token_line = self.current_line; token_col = self.current_col; self.current_col += len(token_value)
                if token_type not in ['COMMENT', 'WHITESPACE']: self.tokens.append((token_type, token_value, token_line, token_col))
        return self.tokens

class Parser:
    """Builds the internal FA representation for DFA and NFA."""
    def __init__(self, tokens):
        self.tokens = tokens; self.current_token_index = 0; self.fa = {'states': set(), 'alphabet': set(), 'start_state': None, 'final': set(), 'transitions': {}}
    def _current_token(self):
        if self.current_token_index < len(self.tokens): return self.tokens[self.current_token_index]
        return None
    def parse(self):
        while self._current_token():
            token_type, token_value, _, _ = self._current_token()
            if token_type == 'KEYWORD':
                if token_value == 'states': self._parse_list('states')
                elif token_value == 'alphabet': self._parse_list('alphabet')
                elif token_value == 'start': self._parse_single_value('start_state')
                elif token_value == 'final': self._parse_list('final')
                elif token_value == 'transitions': self._parse_transitions()
                elif token_value == 'end': break
            else: _, val, line, col = self._current_token(); raise SyntaxError(f"[Line {line}, Col {col}] Unexpected token '{val}'. Expected a keyword.")
        self._validate_semantics(); return self.fa
    def _expect(self, expected_type, expected_value=None):
        token = self._current_token()
        if not token: raise SyntaxError(f"Unexpected end of file. Expected '{expected_type}'.")
        token_type, token_value, line, col = token
        if token_type != expected_type or (expected_value and token_value != expected_value): raise SyntaxError(f"[Line {line}, Col {col}] Expected '{expected_type}' but got '{token_type}:{token_value}'.")
        self.current_token_index += 1; return token_value
    def _parse_list(self, key):
        self._expect('KEYWORD', key); self._expect('COLON')
        while self._current_token() and self._current_token()[0] == 'IDENTIFIER':
            identifier = self._expect('IDENTIFIER'); self.fa[key].add(identifier)
            if self._current_token() and self._current_token()[0] == 'COMMA': self.current_token_index += 1
            else: break
    def _parse_single_value(self, key):
        self._expect('KEYWORD'); self._expect('COLON'); self.fa[key] = self._expect('IDENTIFIER')
    def _parse_transitions(self):
        self._expect('KEYWORD', 'transitions'); self._expect('COLON')
        while self._current_token() and self._current_token()[0] != 'KEYWORD':
            self._expect('LPAREN'); from_state = self._expect('IDENTIFIER'); self._expect('COMMA')
            symbol_token = self._current_token()
            if symbol_token[0] not in ['IDENTIFIER', 'KEYWORD']: raise SyntaxError(f"Expected IDENTIFIER or 'eps' for transition symbol, but got {symbol_token[0]}")
            symbol = symbol_token[1]; self.current_token_index += 1
            self._expect('RPAREN'); self._expect('ARROW')
            to_states = set()
            while self._current_token() and self._current_token()[0] == 'IDENTIFIER':
                to_states.add(self._expect('IDENTIFIER'))
                if self._current_token() and self._current_token()[0] == 'COMMA': self._expect('COMMA')
                else: break
            if not to_states: _, val, line, col = self._current_token(); raise SyntaxError(f"[Line {line}, Col {col}] Expected at least one destination state.")
            if from_state not in self.fa['transitions']: self.fa['transitions'][from_state] = {}
            if symbol not in self.fa['transitions'][from_state]: self.fa['transitions'][from_state][symbol] = set()
            self.fa['transitions'][from_state][symbol].update(to_states)
    def _validate_semantics(self):
        if self.fa['start_state'] and self.fa['start_state'] not in self.fa['states']: raise ValueError(f"Semantic Error: Start state '{self.fa['start_state']}' is not declared.")
        for f_state in self.fa['final']:
            if f_state not in self.fa['states']: raise ValueError(f"Semantic Error: Final state '{f_state}' is not declared.")
        for from_state, transitions in self.fa['transitions'].items():
            if from_state not in self.fa['states']: raise ValueError(f"Semantic Error: Transition state '{from_state}' is not declared.")
            for symbol, to_states in transitions.items():
                if symbol != 'eps' and symbol not in self.fa['alphabet']: raise ValueError(f"Semantic Error: Transition symbol '{symbol}' is not in the alphabet.")
                for to_state in to_states:
                    if to_state not in self.fa['states']: raise ValueError(f"Semantic Error: Transition state '{to_state}' is not declared.")

class Simulator:
    """Runs a simulation and provides a path trace for DFAs and NFAs."""
    def __init__(self, fa):
        self.fa = fa; self.start_state = self.fa.get('start_state'); self.final_states = self.fa.get('final', set()); self.transitions = self.fa.get('transitions', {})
    def is_dfa(self):
        for state_transitions in self.transitions.values():
            if 'eps' in state_transitions: return False
            for symbol_transitions in state_transitions.values():
                if len(symbol_transitions) > 1: return False
        return True
    def run_nfa_with_trace(self, input_string):
        trace = []; active_states = self._epsilon_closure({self.start_state})
        trace.append(('-', active_states.copy()))
        for symbol in input_string:
            if symbol not in self.fa['alphabet']: return f"Rejected (Runtime Error: Input symbol '{symbol}' not in alphabet)", trace, None
            next_active_states = set()
            for state in active_states: next_active_states.update(self.transitions.get(state, {}).get(symbol, set()))
            active_states = self._epsilon_closure(next_active_states)
            trace.append((symbol, active_states.copy()))
            if not active_states: break
        result = "Accepted" if not active_states.isdisjoint(self.final_states) else "Rejected"
        return result, trace, None
    def run_dfa_with_trace(self, input_string):
        trace = []; path = []; current_state = self.start_state
        if not current_state: return "Rejected (Runtime Error: No start state defined)", None, None
        path.append(current_state)
        for symbol in input_string:
            if symbol not in self.fa['alphabet']: return f"Rejected (Runtime Error: Input symbol '{symbol}' not in alphabet)", trace, path
            next_state_set = self.transitions.get(current_state, {}).get(symbol, set())
            if not next_state_set: return "Rejected (Runtime Error: No transition defined)", trace, path
            next_state = next_state_set.pop()
            trace.append((current_state, symbol, next_state))
            path.append(next_state); current_state = next_state
        result = "Accepted" if current_state in self.final_states else "Rejected"
        return result, trace, path
    def _epsilon_closure(self, states):
        closure = set(states); stack = list(states)
        while stack:
            state = stack.pop()
            eps_transitions = self.transitions.get(state, {}).get('eps', set())
            for next_state in eps_transitions:
                if next_state not in closure: closure.add(next_state); stack.append(next_state)
        return closure

class Visualizer:
    """Creates a visual representation for both DFAs and NFAs using Graphviz."""
    def __init__(self, fa):
        self.fa = fa
    def render(self):
        dot = Digraph(comment='Finite Automaton'); dot.attr(rankdir='LR')
        for state in self.fa.get('states', set()):
            shape = 'doublecircle' if state in self.fa.get('final', set()) else 'circle'
            dot.node(state, state, shape=shape)
        start_state = self.fa.get('start_state')
        if start_state: dot.node('', '', shape='none', width='0', height='0'); dot.edge('', start_state, label='')
        transitions = self.fa.get('transitions', {})
        for from_state, trans_map in transitions.items():
            for symbol, to_states in trans_map.items():
                for to_state in to_states:
                    dot.edge(from_state, to_state, label=symbol)
        return dot

#STREAMLIT USER INTERFACE

st.set_page_config(
    layout="wide", 
    page_title="FA Interpreter", 
    initial_sidebar_state="expanded"
)
st.title("Finite Automaton Interpreter")

# Initialize session state
if 'fa_definition' not in st.session_state: st.session_state.fa_definition = None
if 'show_simulation' not in st.session_state: st.session_state.show_simulation = False

#Sidebar (Left Column)
with st.sidebar:
    st.header("1. Define Automaton")
    states_in = st.text_input("States (Q)", "q0,q1,q2,q3", help="Comma-separated state names.")
    alphabet_in = st.text_input("Alphabet (Σ)", "0,1", help="Comma-separated symbols.")
    start_state_in = st.text_input("Start State (q₀)", "q0", help="Single state name.")
    final_states_in = st.text_input("Final States (F)", "q3", help="Comma-separated final state names.")
    transitions_in = st.text_area("Transitions (δ)", "(q0,0)->q0\n(q0,1)->q1\n(q1,0)->q2\n(q1,1)->q0\n(q2,0)->q3\n(q2,1)->q2", height=150, help="One rule per line.")
    
    generate_diagram_button = st.button("Generate Diagram")

    st.markdown("---") #Separator

    st.header("2. Test Sequence")
    input_string = st.text_input("Enter sequence:", "100")
    run_simulation_button = st.button("Run Simulation", disabled=(st.session_state.fa_definition is None))

#Main Area
col_diagram, col_results = st.columns(2)

#Middle Column: Diagram Display
with col_diagram:
    st.subheader("Automaton Structure")
    if generate_diagram_button:
        if not all([states_in, alphabet_in, start_state_in, final_states_in, transitions_in]):
            st.warning("Please fill in all automaton definition fields.")
            st.session_state.fa_definition = None # Reset state if inputs are incomplete
        else:
            fa_code = f"states:{states_in}\nalphabet:{alphabet_in}\nstart:{start_state_in}\nfinal:{final_states_in}\ntransitions:\n{transitions_in}\nend"
            try:
                lexer = Lexer(fa_code); tokens = lexer.tokenize()
                parser = Parser(tokens); fa_definition = parser.parse()
                st.session_state.fa_definition = fa_definition # Store the valid definition
                st.session_state.show_simulation = False # Reset simulation display
                st.success("Definition parsed successfully. Diagram generated.")
            except (SyntaxError, ValueError) as e:
                st.error(f"**Error in Definition:**\n\n{e}"); st.session_state.fa_definition = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}"); st.session_state.fa_definition = None

    # Display the diagram if it has been successfully generated
    if st.session_state.fa_definition:
        visualizer = Visualizer(st.session_state.fa_definition)
        static_graph = visualizer.render()
        st.graphviz_chart(static_graph)
    else:
        st.info("Define the automaton in the sidebar and click 'Generate Diagram'.")

#Right Column: Simulation Results
with col_results:
    st.subheader("Simulation Results")
    if run_simulation_button and st.session_state.fa_definition:
        st.session_state.show_simulation = True # Trigger display

    if st.session_state.show_simulation:
        fa_definition = st.session_state.fa_definition
        simulator = Simulator(fa_definition)
        is_dfa = simulator.is_dfa()

        if is_dfa:
            result, trace, path = simulator.run_dfa_with_trace(input_string)
        else:
            result, trace, _ = simulator.run_nfa_with_trace(input_string)

        #Display final result message
        if "Accepted" in result:
            st.success(f"**Result:** Sequence '{input_string}' is **Accepted**.")
        else:
            st.error(f"**Result:** Sequence '{input_string}' is **Rejected**.")
            if "Runtime Error" in result: st.caption(result)

        #Display Execution Trace
        if trace:
            st.markdown("##### Execution Trace")
            if is_dfa:
                path_data = [{"Step": i + 1, "From State": f, "Input": s, "To State": t} for i, (f, s, t) in enumerate(trace)]
                if path_data:
                    df = pd.DataFrame(path_data); st.table(df.set_index('Step'))
                final_state = path[-1] if path else simulator.start_state
                st.write(f"**End:** Finished in state **`{final_state}`**.")
            else: #NFA Trace
                st.caption("Showing the set of active states after each input symbol.")
                path_data = []
                initial_states_str = ', '.join(sorted(list(trace[0][1])))
                path_data.append({"Step": "Start", "Input Symbol": "ε (initial)", "Active States": f"{{{initial_states_str}}}"})
                for i, (symbol, states) in enumerate(trace[1:]):
                    states_str = ', '.join(sorted(list(states))) if states else '{}'
                    path_data.append({"Step": i + 1, "Input Symbol": f"'{symbol}'", "Active States": f"{{{states_str}}}"})
                df = pd.DataFrame(path_data); st.table(df.set_index('Step'))
                final_states_reached = trace[-1][1]
                st.write(f"**End:** Finished in state set **`{final_states_reached}`**.")
    else:
        st.info("Generate a diagram first, then click 'Run Simulation'.")