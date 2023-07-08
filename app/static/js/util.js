    function zip() {
        for (var i = 0; i < arguments.length; i++) {
            if (!arguments[i].length || !arguments.toString()) {
                return false;
            }
            if (i >= 1) {
                if (arguments[i].length !== arguments[i - 1].length) {
                    return false;
                }
            }
        }
        var zipped = [];
        for (var j = 0; j < arguments[0].length; j++) {
            var toBeZipped = [];
            for (var k = 0; k < arguments.length; k++) {
                toBeZipped.push(arguments[k][j]);
            }
            zipped.push(toBeZipped);
        }
        return zipped;
    }

    function get_diff_ratio(s_old, s_new){
        var s = new difflib.SequenceMatcher(null, s_old, s_new)
        return s.ratio()
    }

    function get_word_diff_ratio(s_old, s_new) {
        var s_old_words = s_old.split(' ')
        var s_new_words = s_new.split(' ')
        var s = new difflib.SequenceMatcher(null, s_old_words, s_new_words)
        return s.ratio()
    }

    function get_list_diff(l_old, l_new){
        var vars_old = []
        var vars_new = []
        var diffs = difflib.ndiff(l_old, l_new)
        var in_question = false
        diffs.forEach(function(item, idx){
            var label = item[0]
            var text = item.slice(2)
            if (label == '?'){
                return
            }

            else if (label == '-'){
                vars_old.push({
                    'text': text,
                    'tag': '-'
                })
                if (
                        // if something is removed from the old sentence, a '?' will be present in the next idx
                        ((idx < diffs.length - 1) && (diffs[idx + 1][0] == '?'))
                        // if NOTHING is removed from the old sentence, a '?' might still be present in 2 idxs, unless the next sentence is a - as well.
                     || ((idx < diffs.length - 2) && (diffs[idx + 2][0] == '?') && (diffs[idx + 1][0] != '-'))
                ){
                    in_question = true
                    return
                }
                // test if the sentences are substantially similar, but for some reason ndiff marked them as different.
                if ((idx < (diffs.length - 1)) && (diffs[idx + 1][0] == '+')){
                    var text_new = diffs[idx + 1].slice(2)
                    if (get_word_diff_ratio(text, text_new) > .8) {
                        in_question = true
                        return
                    }
                }
                vars_new.push({
                    'text': '',
                    'tag': ' '
                })
            }
            else if (label == '+'){
                vars_new.push({
                    'text': text,
                    'tag': '+'
                })
                if (in_question){
                    in_question = false
                }
                else{
                    vars_old.push({
                        'text':'',
                        'tag': ' '
                    })
                }
            }
            else {
                vars_old.push({
                    'text': text,
                    'tag': ' '
                })
                vars_new.push({
                    'text': text,
                    'tag': ' '
                })
            }
        })
        return [vars_old, vars_new]
    }

    function get_word_diffs(s_old, s_new) {
        var s_old_words = s_old.split(' ')
        var s_new_words = s_new.split(' ')
        return get_list_diff(s_old_words, s_new_words)
    }

    function html_compare_sentences(old_sent, new_sent) {
        var sents = get_word_diffs(old_sent, new_sent)
        old_sent = sents[0]
        new_sent = sents[1]
        var new_html = []
        var old_html = []
        var max_idx = Math.max(old_sent.length, new_sent.length)
        for (var idx = 0; idx < max_idx; idx++) {
            var w_old = old_sent[idx]
            var w_new = new_sent[idx]
            if (w_old['tag'] == '-') {
                old_html.push('<span class="deleted">' + w_old['text'] + '</span>')
            } else {
                old_html.push(w_old['text'])
            }
            if (w_new['tag'] == '+') {
                new_html.push('<span class="added">' + w_new['text'] + ' </span>')
            } else {
                new_html.push(w_new['text'])
            }
        }
        return [old_html.join(' '), new_html.join(' ')]
    }

    String.prototype.replaceAll = function(search, replacement) {
        var target = this;
        return target.replace(new RegExp(search, 'g'), replacement);
    };
    String.prototype.toTitleCase = function() {
        var target = this;
        return target.replace(/(?:^|\s)\w/g, function(match) {
            return match.toUpperCase();
        });
    };
    Array.prototype.remove_duplicates = function() {
        var arr = this;
        let s = new Set(arr);
        let it = s.values();
        return Array.from(it);
    };
    // Array.prototype.remove_one_by_value = function(value) {
    //     var index = this.indexOf(value);
    //     if (index > -1) {
    //         this.splice(index, 1);
    //     }
    //     return this;
    // }
    Array.prototype.remove_one_by_value = function(val) {
      for (var i = 0; i < this.length; i++) {
        if (this[i].toString() === val.toString()) {
          this.splice(i, 1);
          break
        }
      }
      return this;
    };
    Array.prototype.remove_by_value = function(val) {
      for (var i = 0; i < this.length; i++) {
        if (this[i] === val) {
          this.splice(i, 1);
          i = i - 1;
        }
      }
      return this;
    };