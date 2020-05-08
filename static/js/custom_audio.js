$(document).ready(function(){
    (function($) {
        $.ustamp = function() {
	    return 'xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
	        var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
		return v.toString(16);
	    });
	};
    })(jQuery);

    var audio_upload_url  = '/audio';
    $.extend({ alert: function (message, title) {
        $("<div></div>").dialog( {
            buttons: { "Ok": function () { $(this).dialog("close"); } },
            close: function (event, ui) { $(this).remove(); },
            resizable: false,
            title: title,
            modal: true
        }).text(message);
    }});

    function alertNew(message){
        $.alert(message, message);
    }

    function createResultsTable(result){
		console.log('****************')
		console.log(result);
		var new_lnk = document.createElement('table');
		new_lnk.className = "table table-responsive table-condensed table-hover"
        
		$(new_lnk).attr('rules', 'groups');

		table_html = '</caption><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>';
		for (var value of result){
				table_html = table_html + '<tr><td>' + value[0] + '</td><td>' + value[1] + '</td></tr>';
		}
		table_html = table_html + '</tbody></table>';

		//$('tbody').sortable();
		new_lnk.innerHTML = table_html;
		return new_lnk;
	}

	function addEndResults(data){
        var end_results_ele = document.getElementById('id_end_results');
        var response = JSON.parse(data)['response'];
		var result  = response[0]['scores'];

        var half_column = document.createElement('div')
        half_column.className = "form-group col-lg-6";
		//half_column.innerHTML = '<h5>Predicted Class : ' + prediction + '</h5>';

        end_results_ele.append(half_column);

		// create results table
    	half_column.append(createResultsTable(result));
	}

    $("#submit_predict").click(function(){
	    var image_id_list     = [];

		// Clear id_results
	    $('#id_end_results').empty();

	    $('#id_results').css('overflow-y', 'scroll');

	    // Stage 0 processing function
	    function processStage0(){
                var imgs_list  = document.getElementById('id_audio').files;
		var e_i = false;

		if (imgs_list.length == 0){
		    alertNew('No image uploaded !!');
	            e_i = true;
		}

		// if any of the video list or image list is empty don't proceed
		if (!(e_i)){
		    processStage1();
		}
	    }

	    // This is Stage 1 function
	    // It takes care of uploading the image
	    function processStage1(){
		    var image_file = document.getElementById('id_audio').files[0];
		    var formdata   = new FormData();
		    formdata.append('audio', image_file);

            // Call ajax
            $.ajax({
                // Your server script to process the upload
                url            : audio_upload_url,
                type           : 'POST',
                // Form data
                data           : formdata,
                // Tell jQuery not to process data or worry about content-type
                // You *must* include these options!
                cache          : false,
                contentType    : false,
                processData    : false,
            	        
            }).done(function(data){
		    			console.log(data);
		    			addEndResults(data);
		    });
        }

	    processStage0();

    });

});
