"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { uploadPDF, UploadResponse } from "../../lib/api";

interface Props {
    onUploadSuccess: (data: UploadResponse) => void;
}

export default function PDFUpload({ onUploadSuccess }: Props) {
    const [status, setStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");
    const [message, setMessage] = useState("");

    const onDrop = useCallback(async (accepted: File[]) => {
        const file = accepted[0];
        if (!file) return;

        setStatus("uploading");
        setMessage("");

        try {
            const data = await uploadPDF(file);
            setStatus("success");
            setMessage(`${data.num_chunks} chunks · ${data.num_questions} questions generated`);
            onUploadSuccess(data);
        } catch (err: any) {
            setStatus("error");
            setMessage(err.message || "Upload failed");
        }
    }, [onUploadSuccess]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { "application/pdf": [".pdf"] },
        maxFiles: 1,
        disabled: status === "uploading",
    });

    return (
        <div className="w-full max-w-xl mx-auto">
            <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors
          ${isDragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400"}
          ${status === "uploading" ? "opacity-60 cursor-not-allowed" : ""}
        `}
            >
                <input {...getInputProps()} />

                {status === "uploading" ? (
                    <p className="text-gray-500 text-sm">Uploading and processing...</p>
                ) : isDragActive ? (
                    <p className="text-blue-500 font-medium">Drop the PDF here</p>
                ) : (
                    <>
                        <p className="text-gray-600 font-medium">Drag & drop a PDF here</p>
                        <p className="text-gray-400 text-sm mt-1">or click to browse</p>
                    </>
                )}
            </div>

            {status === "success" && (
                <p className="mt-3 text-green-600 text-sm text-center">✅ {message}</p>
            )}
            {status === "error" && (
                <p className="mt-3 text-red-500 text-sm text-center">❌ {message}</p>
            )}
        </div>
    );
}