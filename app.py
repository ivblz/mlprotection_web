import os
import sys
import io
import uuid

from flask import Flask, request, render_template, send_file, url_for, redirect, session
import pandas as pd
import CoreWebVersion
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ваш_секретный_случайный_ключ')

analysis_results = {}


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)

                task_type = request.form.get('task_type')
                visualize = 'visualize' in request.form

                classification_param = False
                if task_type == 'classification':
                    classification_param = True

                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                session_id = uuid.uuid4().hex[:12]
                warning_file_name = f"warning_data_{session_id}.csv"
                treated_file_name = f"treat_data_{session_id}.csv"

                warning_df, treated_df, plot_bytes_list = CoreWebVersion.start(
                    df,
                    classification=classification_param,
                    visualize=visualize,
                    plot_list_to_append=[]
                )

                analysis_results[session_id] = {
                    'warning': warning_df,
                    'treated': treated_df,
                    'warning_file': warning_file_name,
                    'treated_file': treated_file_name,
                    'plots': plot_bytes_list,
                    'original_filename': file.filename,
                    'terminal_output': captured_output.getvalue()
                }
                session['session_id'] = session_id

            except Exception as e:
                sys.stdout = old_stdout
                return render_template('index.html', error_message=f"Ошибка при анализе файла: {str(e)}")
            finally:
                sys.stdout = old_stdout

            return render_template(
                'index.html',
                terminal_output=analysis_results[session_id]['terminal_output'],
                warning_file=warning_file_name,
                treated_file=treated_file_name,
                plots=range(len(analysis_results[session_id]['plots'])),
                original_filename=file.filename,
                results_exist=True
            )

    return render_template('index.html', results_exist=False)


@app.route('/download/results/<filename>')
def download_file(filename):
    session_id = session.get('session_id')
    if not session_id or session_id not in analysis_results:
        return "Нет данных для скачивания", 404
    result = analysis_results[session_id]
    if filename == result['warning_file']:
        df = result['warning']
    elif filename == result['treated_file']:
        df = result['treated']
    else:
        return "Файл не найден", 404
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name=filename)


@app.route('/plots/<int:plot_idx>')
def serve_plot(plot_idx):
    session_id = session.get('session_id')
    if not session_id or session_id not in analysis_results:
        return "Нет данных для графика", 404
    plots = analysis_results[session_id].get('plots', [])
    if plot_idx < 0 or plot_idx >= len(plots):
        return "График не найден", 404
    return send_file(io.BytesIO(plots[plot_idx]), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)