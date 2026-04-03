"use client";

import { AskResponse } from "../../lib/api";

interface Props {
    response: AskResponse | null;
}

export default function ModelCompare({ response }: Props) {
    if (!response) return null;

    const { bert, distilbert } = response;

    const ConfidenceBar = ({ value }: { value: number }) => (
        <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1">
            <div
                className="bg-blue-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.round(value * 100)}%` }}
            />
        </div>
    );

    const ModelCard = ({
        label, color, result
    }: {
        label: string;
        color: string;
        result: typeof bert;
    }) => (
        <div className={`flex-1 border ${color} rounded-xl p-4`}>
            <p className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                {label}
            </p>
            <p className="text-gray-800 font-medium text-sm mb-3">
                {result.answer}
            </p>
            <div className="text-xs text-gray-500 space-y-1">
                <div>
                    <span>Confidence: {Math.round(result.confidence * 100)}%</span>
                    <ConfidenceBar value={result.confidence} />
                </div>
                <p>Latency: {result.latency_ms} ms</p>
            </div>
        </div>
    );

    return (
        <div className="w-full max-w-xl mx-auto mt-6">
            <p className="text-sm font-medium text-gray-600 mb-3">Model Comparison</p>

            <div className="flex gap-3">
                <ModelCard
                    label="BERT"
                    color="border-blue-200"
                    result={bert}
                />
                <ModelCard
                    label="DistilBERT"
                    color="border-orange-200"
                    result={distilbert}
                />
            </div>

            {/* Winner badge */}
            <div className="mt-3 flex gap-3 text-xs text-gray-500">
                <div className="flex-1 text-center">
                    {bert.confidence > distilbert.confidence && (
                        <span className="bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
                            Higher confidence ✓
                        </span>
                    )}
                </div>
                <div className="flex-1 text-center">
                    {distilbert.latency_ms < bert.latency_ms && (
                        <span className="bg-orange-100 text-orange-600 px-2 py-0.5 rounded-full">
                            Faster response ✓
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}