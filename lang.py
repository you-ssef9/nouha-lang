#!/usr/bin/env python3
import sys
import math

variables = {}
functions = {}

def say(msg):
    print(msg)

def set_var(var, value):
    try:
        variables[var] = eval(value, {}, variables)
    except:
        variables[var] = value.strip('"').strip("'")

def eval_expr(expr):
    try:
        return eval(expr, {}, variables)
    except:
        return expr.strip('"').strip("'")

def run_block(lines):
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        elif line.startswith("say "):
            say(eval_expr(line[4:]))
        elif line.startswith("set "):
            parts = line[4:].split("=")
            set_var(parts[0].strip(), parts[1].strip())
        elif line.startswith("if "):
            cond = line[3:].rstrip(":")
            block = []
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            if eval_expr(cond):
                run_block(block)
            continue
        elif line.startswith("else:"):
            block = []
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            run_block(block)
            continue
        elif line.startswith("loop "):
            parts = line[5:].split(" in ")
            var = parts[0].strip()
            start_end = parts[1].split("..")
            start = int(eval_expr(start_end[0].strip()))
            end = int(eval_expr(start_end[1].strip()))
            block = []
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            for n in range(start, end+1):
                variables[var] = n
                run_block(block)
            continue
        elif line.startswith("def "):
            func_name = line[4:].split("(")[0].strip()
            func_args = line[line.find("(")+1:line.find(")")].split(",")
            func_args = [arg.strip() for arg in func_args if arg.strip()]
            block = []
            i += 1
            while i < len(lines) and lines[i].startswith("    "):
                block.append(lines[i][4:])
                i += 1
            functions[func_name] = (func_args, block)
            continue
        elif line.startswith("use "):
            lib = line[4:].strip()
            if lib == "math":
                variables["math"] = math
        elif "(" in line and ")" in line:
            # call function
            func_name = line[:line.find("(")].strip()
            args = line[line.find("(")+1:line.find(")")].split(",")
            args = [eval_expr(a.strip()) for a in args]
            if func_name in functions:
                func_args, block = functions[func_name]
                old_vars = variables.copy()
                for a,v in zip(func_args,args):
                    variables[a] = v
                run_block(block)
                variables.update(old_vars)
        i += 1

def main():
    if len(sys.argv) < 2:
        print("Usage: nouha file.nh")
        return
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    run_block(lines)

if __name__ == "__main__":
    main()
