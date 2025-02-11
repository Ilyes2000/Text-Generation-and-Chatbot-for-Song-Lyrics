import gradio as gr
import json
import os
import zipfile

def load_data(filepath):
    """Loads data from the JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def create_comparison_app(file_paths):
    """Creates the Gradio app for comparing LLM responses with side-by-side layout for multiple files and browser download."""

    all_data = {}  # Dictionary to store data for each file, keyed by filepath
    current_file_index = 0
    current_prompt_index = 0
    current_filepath = ""
    results_data = {} # store results data in memory

    def initialize_data(filepath):
        nonlocal all_data, current_prompt_index, current_filepath, results_data
        if filepath not in all_data:
            all_data[filepath] = load_data(filepath)
            results_data[filepath] = list(all_data[filepath]) # Create a copy to store results, important to not modify original data in all_data directly
        current_filepath = filepath
        current_prompt_index = 0

    def get_progress_text():
        nonlocal current_file_index, file_paths, current_prompt_index
        files_left = len(file_paths) - (current_file_index + 1)
        if current_filepath:
            prompts_left = len(results_data[current_filepath]) - (current_prompt_index + 1) if current_prompt_index < len(results_data[current_filepath]) else 0
            return f"File {current_file_index + 1}/{len(file_paths)} - {prompts_left + 1} prompts left in this file, {files_left} files remaining."
        else:
            return "No file loaded."

    def display_prompt_and_responses(filepath, index):
        """Displays the prompt and responses for a given index within the current file."""
        if not filepath or filepath not in results_data: # Use results_data here
            return "No file loaded.", "", "", get_progress_text(), None

        data = results_data[filepath] # Use results_data
        if 0 <= index < len(data):
            item = data[index]
            prompt_text = item.get("prompt", "No prompt available")
            finetuned_output_text = item.get("finetuned_output", "No finetuned output")
            base_output_text = item.get("base_output", "No base output")
            return prompt_text, finetuned_output_text, base_output_text, get_progress_text(), None # None for file download initially
        else:
            return "File finished! Please proceed to the next file.", "", "", get_progress_text(), None # Indicate file completion, None for file download

    def record_choice(choice):
        """Records the user's choice and moves to the next prompt or file, provides download at the end."""
        nonlocal current_prompt_index, results_data, current_filepath, current_file_index, file_paths

        if not current_filepath:
            return "No file loaded.", "", "", get_progress_text(), None

        data = results_data[current_filepath] # Use results_data
        if 0 <= current_prompt_index < len(data):
            if choice == "finetuned":
                data[current_prompt_index]["choice"] = "finetuned"
            elif choice == "base":
                data[current_prompt_index]["choice"] = "base_output"

            current_prompt_index += 1
            if current_prompt_index < len(data):
                return display_prompt_and_responses(current_filepath, current_prompt_index) + (None,) # None for file download
            else:
                # File finished, prepare for download
                if current_file_index < len(file_paths) - 1:
                    current_file_index += 1
                    next_filepath = file_paths[current_file_index]
                    initialize_data(next_filepath) # Initialize for the next file
                    return display_prompt_and_responses(current_filepath, current_prompt_index) + (None,) # None for file download, start next
                else:
                    # All files finished, prepare zip archive for download
                    zip_filepath = create_zip_archive(results_data)
                    return "Comparison finished for all files! Please download the results ('Download results' button).", "", "", "Comparison finished for all files!", gr.update(visible=True, value=zip_filepath, label="Download All Results") # Final completion, with file download
        else:
            # Should not reach here normally, but handle for robustness - in case record_choice is called after file is finished
            if current_file_index < len(file_paths) - 1:
                current_file_index += 1
                next_filepath = file_paths[current_file_index]
                initialize_data(next_filepath) # Initialize for the next file
                return display_prompt_and_responses(current_filepath, current_prompt_index) + (None,) # None for file download, start next
            else:
                # All files finished, prepare zip archive for download
                zip_filepath = create_zip_archive(results_data)
                return "Comparison finished for all files! Please download the results ('Download results' button).", "", "", "Comparison finished for all files!", gr.update(visible=True, value=zip_filepath, label="Download All Results")

    def create_zip_archive(results_data):
        """Creates a zip archive of all result files."""
        zip_filepath = "/tmp/results.zip"
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            for filepath, data in results_data.items():
                results_filename = os.path.basename(filepath).replace(".json", "_results.json")
                results_json_string = json.dumps(data, indent=2)
                zipf.writestr(results_filename, results_json_string)
        return zip_filepath

    with gr.Blocks() as iface:
        progress_markdown = gr.Markdown(get_progress_text()) # Progress indication at the top
        gr.Markdown("# LLM song lyrics generation ranking")
        prompt_output = gr.Textbox(label="Lyrics description", lines=3, interactive=False, max_lines=3) # Fixed lines for prompt

        with gr.Row(): # Row for side-by-side outputs
            with gr.Column(): # Column for Finetuned Output
                finetuned_output_box = gr.Textbox(label="Model A", lines=10, interactive=False, max_lines=10) # Fixed lines for finetuned
            with gr.Column(): # Column for Base Output
                base_output_box = gr.Textbox(label="Model B", lines=10, interactive=False, max_lines=10) # Fixed lines for base

        with gr.Row(): # Row for buttons
            finetuned_button = gr.Button("Model A is better")
            base_button = gr.Button("Model B is better")

        file_download_output = gr.DownloadButton(label="Download results", visible=False) # File output component

        def load_initial_file(files):
            if files:
                filepath = files[0] # Take the first file path from the list
                initialize_data(filepath)
                return display_prompt_and_responses(current_filepath, current_prompt_index) + (None,) # None for initial file download
            return "No file loaded.", "", "", get_progress_text(), None # None for initial file download

        # Initial display - needs to load the first file
        iface.load(
            load_initial_file,
            inputs=[gr.State(file_paths)], # Pass the list of filepaths as state
            outputs=[prompt_output, finetuned_output_box, base_output_box, progress_markdown, file_download_output] # Added file_download_output
        )

        # Button click events
        finetuned_button.click(
            fn=record_choice,
            inputs=[gr.State("finetuned")], # Pass "finetuned" string
            outputs=[prompt_output, finetuned_output_box, base_output_box, progress_markdown, file_download_output], # Added file_download_output
            api_name="choose_finetuned"
        )
        base_button.click(
            fn=record_choice,
            inputs=[gr.State("base")], # Pass "base" string
            outputs=[prompt_output, finetuned_output_box, base_output_box, progress_markdown, file_download_output], # Added file_download_output
            api_name="choose_base"
        )

    return iface

if __name__ == '__main__':
    json_files = ["./Qwen2.5-0.5B.json", "./SmolLM135.json", "./SmolLM135-instruct.json", "./SmolLM360.json", "./SmolLM360-instruct.json"]
    app = create_comparison_app(json_files)
    app.launch()
