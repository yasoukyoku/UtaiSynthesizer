const fs = require('fs');
const path = require('path');

const newKeys = {
  exampleLibrary: {
    title: { zh: "示例工作流库", en: "Example Workflow Library", ja: "サンプルワークフロー集" },
    button: { zh: "示例", en: "Examples", ja: "サンプル" },
    loaded: { zh: "已加载示例工作流:", en: "Example workflow loaded:", ja: "サンプルワークフローを読み込みました:" },
    allCategories: { zh: "全部类别", en: "All Categories", ja: "全カテゴリ" },
    categoryBasic: { zh: "基础流程", en: "Basic Workflows", ja: "基本フロー" },
    categoryAdvanced: { zh: "高级应用", en: "Advanced Usage", ja: "高度な利用" },
    categoryEffect: { zh: "效果处理", en: "Effects", ja: "エフェクト処理" },
    categoryAnalysis: { zh: "分析工具", en: "Analysis Tools", ja: "解析ツール" },
    allDifficulties: { zh: "全部难度", en: "All Difficulties", ja: "全難易度" },
    difficultyBeginner: { zh: "新手入门", en: "Beginner", ja: "初心者向け" },
    difficultyIntermediate: { zh: "进阶使用", en: "Intermediate", ja: "中級者向け" },
    difficultyAdvanced: { zh: "专业级", en: "Advanced", ja: "上級者向け" },
    filterCategory: { zh: "类别", en: "Category", ja: "カテゴリ" },
    filterDifficulty: { zh: "难度", en: "Difficulty", ja: "難易度" },
    nodes: { zh: "个节点", en: " nodes", ja: " ノード" },
    loadButton: { zh: "加载", en: "Load", ja: "読み込む" },
    noResults: { zh: "没有符合条件的示例工作流", en: "No matching example workflows", ja: "条件に合うサンプルワークフローがありません" }
  },
  annotation: {
    title: { zh: "节点注释", en: "Node Annotation", ja: "ノード注釈" },
    add: { zh: "添加注释", en: "Add Annotation", ja: "注釈を追加" },
    edit: { zh: "编辑注释", en: "Edit Annotation", ja: "注釈を編集" },
    placeholder: { zh: "在此输入注释内容...", en: "Enter annotation text...", ja: "注釈内容を入力..." },
    colorLabel: { zh: "颜色", en: "Color", ja: "カラー" },
    colorYellow: { zh: "黄色", en: "Yellow", ja: "イエロー" },
    colorBlue: { zh: "蓝色", en: "Blue", ja: "ブルー" },
    colorGreen: { zh: "绿色", en: "Green", ja: "グリーン" },
    colorRed: { zh: "红色", en: "Red", ja: "レッド" },
    colorPurple: { zh: "紫色", en: "Purple", ja: "パープル" },
    deleteButton: { zh: "删除注释", en: "Delete Annotation", ja: "注釈を削除" },
    cancelButton: { zh: "取消", en: "Cancel", ja: "キャンセル" },
    saveButton: { zh: "保存", en: "Save", ja: "保存" }
  },
  portKind: {
    audio: { zh: "音频", en: "Audio", ja: "オーディオ" },
    midi: { zh: "MIDI", en: "MIDI", ja: "MIDI" },
    chords: { zh: "和弦", en: "Chords", ja: "コード" },
    lyrics: { zh: "歌词", en: "Lyrics", ja: "歌詞" },
    report: { zh: "报告", en: "Report", ja: "レポート" },
    any: { zh: "任意", en: "Any", ja: "任意" }
  },
  smartConnection: {
    compatibleCount: { zh: "{{count}} 个兼容端口", en: "{{count}} compatible ports", ja: "{{count}} 個の互換ポート" },
    noCompatible: { zh: "无兼容端口", en: "No compatible ports", ja: "互換ポートなし" }
  },
  tutorial: {
    skipButton: { zh: "跳过引导", en: "Skip Tutorial", ja: "スキップ" },
    previousButton: { zh: "上一步", en: "Previous", ja: "前へ" },
    nextButton: { zh: "下一步", en: "Next", ja: "次へ" },
    completeButton: { zh: "完成", en: "Complete", ja: "完了" },
    step1: {
      title: { zh: "欢迎使用工作流编辑器", en: "Welcome to the Workflow Editor", ja: "ワークフローエディタへようこそ" },
      description: { zh: "这是节点调色板，包含所有可用的处理节点。从这里拖动节点到画布即可开始构建工作流。", en: "This is the node palette containing all available processing nodes. Drag nodes from here to the canvas to start building your workflow.", ja: "これはノードパレットで、利用可能なすべての処理ノードが含まれています。ここからノードをキャンバスにドラッグしてワークフローの構築を始めましょう。" }
    },
    step2: {
      title: { zh: "连接节点", en: "Connecting Nodes", ja: "ノード接続" },
      description: { zh: "将鼠标悬停在节点上，可以看到输入输出端口。拖动端口之间的连线来构建处理流程。", en: "Hover over a node to see its input and output ports. Drag connections between ports to build your processing flow.", ja: "ノードの上にマウスをホバーすると、入力と出力ポートが表示されます。ポート間をドラッグして処理フローを構築しましょう。" },
      action: { zh: "试着连接两个节点", en: "Try connecting two nodes", ja: "2つのノードを接続してみましょう" }
    },
    step3: {
      title: { zh: "配置参数", en: "Configure Parameters", ja: "パラメータ設定" },
      description: { zh: "点击节点可以查看和调整参数。每个节点都有详细的参数说明和单位显示。", en: "Click on a node to view and adjust its parameters. Each node has detailed parameter descriptions and unit displays.", ja: "ノードをクリックしてパラメータを表示・調整できます。各ノードには詳細なパラメータ説明と単位表示があります。" }
    },
    step4: {
      title: { zh: "运行工作流", en: "Run Workflow", ja: "ワークフロー実行" },
      description: { zh: "连接好节点后，点击顶部的运行按钮即可执行整个工作流。你也可以点击单个节点上的运行按钮来单独执行。", en: "Once nodes are connected, click the run button at the top to execute the entire workflow. You can also click the run button on individual nodes to execute them separately.", ja: "ノードを接続したら、上部の実行ボタンをクリックしてワークフロー全体を実行できます。個別のノードの実行ボタンをクリックして単独実行もできます。" }
    },
    step5: {
      title: { zh: "更多功能", en: "More Features", ja: "その他の機能" },
      description: { zh: "右键点击节点或画布可以打开上下文菜单，查看更多操作选项。使用这里的控制按钮可以调整视图。", en: "Right-click on nodes or the canvas to open the context menu and see more options. Use the control buttons here to adjust your view.", ja: "ノードまたはキャンバスを右クリックしてコンテキストメニューを開き、さらなるオプションを表示できます。ここのコントロールボタンでビューを調整できます。" }
    }
  },
  error: {
    suggestionsTitle: { zh: "建议解决方案", en: "Suggested Solutions", ja: "推奨解決策" },
    technicalDetails: { zh: "技术详情", en: "Technical Details", ja: "技術詳細" },
    severityError: { zh: "错误", en: "Error", ja: "エラー" },
    severityWarning: { zh: "警告", en: "Warning", ja: "警告" },
    severityInfo: { zh: "提示", en: "Info", ja: "情報" },
    suggestion: {
      modelNotFound: {
        title: { zh: "未找到模型文件", en: "Model File Not Found", ja: "モデルファイルが見つかりません" },
        description: { zh: "请确认模型已正确下载并转换。", en: "Please ensure the model has been correctly downloaded and converted.", ja: "モデルが正しくダウンロードおよび変換されていることを確認してください。" },
        action: { zh: "前往模型管理", en: "Go to Model Manager", ja: "モデル管理へ" }
      },
      gpuMemory: {
        title: { zh: "GPU 显存不足", en: "Insufficient GPU Memory", ja: "GPU メモリ不足" },
        description: { zh: "尝试降低批处理大小，或关闭其他占用显存的程序。", en: "Try reducing the batch size or closing other programs using GPU memory.", ja: "バッチサイズを減らすか、他のGPUメモリを使用するプログラムを終了してください。" }
      },
      fileNotFound: {
        title: { zh: "文件未找到", en: "File Not Found", ja: "ファイルが見つかりません" },
        description: { zh: "请检查文件路径是否正确，文件是否已被移动或删除。", en: "Please check if the file path is correct and if the file has been moved or deleted.", ja: "ファイルパスが正しいか、ファイルが移動または削除されていないか確認してください。" }
      },
      connection: {
        title: { zh: "网络连接问题", en: "Network Connection Issue", ja: "ネットワーク接続の問題" },
        description: { zh: "请检查网络连接，或稍后重试。", en: "Please check your network connection or try again later.", ja: "ネットワーク接続を確認するか、後でもう一度お試しください。" }
      },
      voiceIndex: {
        title: { zh: "索引文件问题", en: "Index File Issue", ja: "インデックスファイルの問題" },
        description: { zh: "检查 index_ratio 参数设置，或重新训练模型的索引文件。", en: "Check the index_ratio parameter setting or retrain the model's index file.", ja: "index_ratio パラメータの設定を確認するか、モデルのインデックスファイルを再トレーニングしてください。" }
      },
      generic: {
        title: { zh: "常规解决方案", en: "General Solution", ja: "一般的な解決策" },
        description: { zh: "尝试重新运行，或查看日志了解详细错误信息。", en: "Try running again or check the logs for detailed error information.", ja: "再実行するか、ログで詳細なエラー情報を確認してください。" }
      }
    }
  }
};

function addNestedKeys(target, source, lang) {
  Object.keys(source).forEach(key => {
    if (typeof source[key] === 'object' && source[key][lang]) {
      target[key] = source[key][lang];
    } else if (typeof source[key] === 'object') {
      if (!target[key]) target[key] = {};
      addNestedKeys(target[key], source[key], lang);
    }
  });
}

['en', 'ja'].forEach(lang => {
  const filePath = path.join(__dirname, 'src', 'i18n', `${lang}.json`);
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  
  if (!data.workflow.exampleLibrary) {
    data.workflow.exampleLibrary = {};
  }
  addNestedKeys(data.workflow.exampleLibrary, newKeys.exampleLibrary, lang);
  
  if (!data.workflow.annotation) {
    data.workflow.annotation = {};
  }
  addNestedKeys(data.workflow.annotation, newKeys.annotation, lang);
  
  if (!data.workflow.portKind) {
    data.workflow.portKind = {};
  }
  addNestedKeys(data.workflow.portKind, newKeys.portKind, lang);
  
  if (!data.workflow.smartConnection) {
    data.workflow.smartConnection = {};
  }
  addNestedKeys(data.workflow.smartConnection, newKeys.smartConnection, lang);
  
  if (!data.workflow.tutorial) {
    data.workflow.tutorial = {};
  }
  addNestedKeys(data.workflow.tutorial, newKeys.tutorial, lang);
  
  if (!data.workflow.error) {
    data.workflow.error = {};
  }
  addNestedKeys(data.workflow.error, newKeys.error, lang);
  
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
  console.log(`Updated ${lang}.json`);
});

console.log('Translation update completed!');
