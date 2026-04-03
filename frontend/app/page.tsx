"use client";

import { useState } from "react";
import PDFUpload from "./components/PDFUpload";
import AutoQA from "./components/AutoQA";
import ChatBox from "./components/ChatBox";
import ModelCompare from "./components/ModelCompare";
import { AskResponse, UploadResponse } from "../lib/api";

export default function Home() {
  const [uploaded, setUploaded] = useState(false);
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [prefillQ, setPrefillQ] = useState("");
  const [latestResponse, setLatestResponse] = useState<AskResponse | null>(null);

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadData(data);
    setUploaded(true);
    setLatestResponse(null);
    setPrefillQ("");
  };

  return (
    <main className="min-h-screen bg-white px-4 py-10">
      <div className="max-w-xl mx-auto">

        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-bold text-gray-800">DocQuery</h1>
          <p className="text-gray-400 text-sm mt-1">
            Upload a PDF and ask questions — answered by BERT and DistilBERT
          </p>
        </div>

        {/* Upload */}
        <PDFUpload onUploadSuccess={handleUploadSuccess} />

        {/* Upload stats */}
        {uploadData && (
          <p className="text-center text-xs text-gray-400 mt-2">
            {uploadData.num_chunks} chunks loaded
          </p>
        )}

        {/* Suggested questions */}
        <AutoQA
          triggered={uploaded}
          onQuestionClick={(q) => setPrefillQ(q)}
        />

        {/* Chat */}
        {uploaded && (
          <ChatBox
            prefillQuestion={prefillQ}
            onNewResponse={(r) => setLatestResponse(r)}
          />
        )}

        {/* Model comparison */}
        <ModelCompare response={latestResponse} />

      </div>
    </main>
  );
}