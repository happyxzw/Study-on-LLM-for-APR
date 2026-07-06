private static void handleRawtext(Token.StartTag startTag, HtmlTreeBuilder tb) {
    tb.insert(startTag); // Buggy Line
    tb.tokeniser.transition(TokeniserState.Rawtext);
    tb.markInsertionMode();
    tb.transition(Text);
}

// End


// Buggy Function
function foo() {
  console.log("This is a log statement");
  bar();
}

function bar() {
  console.log("This is another log statement");
}

foo();

// Fixed Function
function foo() {
  console.log("This is a log statement");
  bar();
}