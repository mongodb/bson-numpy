"use strict";

function warnProjectDeprecated() {
  var warning = document.createElement('div');
  warning.setAttribute('class', 'admonition danger');
  warning.innerHTML = "<p class='first admonition-title'>Attention</p> " +
    "<p class='last'> " +
    "BSON-NumPy has been superseded by <a href='https://mongo-arrow.readthedocs.io/'>PyMongoArrow</a> " +
    "and is no longer actively developed. In addition to NumPy arrays, <strong>PyMongoArrow</strong> " +
    "also supports direct conversion of MongoDB query results to Pandas DataFrames and Apache Arrow Tables. " +
    "Users are encouraged to migrate their BSON-NumPy workloads to PyMongoArrow for continued support." +
    "</p>";

  var parent = document.querySelector('div.body')
    || document.querySelector('div.document')
    || document.body;
  parent.insertBefore(warning, parent.firstChild);
}

document.addEventListener('DOMContentLoaded', warnProjectDeprecated);
