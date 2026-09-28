'use client';
import React, { useState, useMemo, useEffect } from 'react';
import { LevelHistory, HistoryEntry } from '../types';

interface HistoryExplorerProps {
  levelsHistory: LevelHistory[];
  selectedLevelId: string | null;
  selectedHistoryId: string | null;
  historyEntries: HistoryEntry[];
  historyLoading: boolean;
  selectLevel: (levelId: string) => void;
  selectHistoryEntry: (historyId: string) => void;
  isHistoricalView: boolean;
  returnToLive: () => void;
}

export const HistoryExplorer: React.FC<HistoryExplorerProps> = ({
  levelsHistory,
  selectedLevelId,
  selectedHistoryId,
  historyEntries,
  historyLoading,
  selectLevel,
  selectHistoryEntry,
  isHistoricalView,
  returnToLive
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  // Reset category when level changes
  useEffect(() => {
    setSelectedCategory(null);
  }, [selectedLevelId]);

  const categories = ['INITIAL', 'ROLLBACK', 'ADAPT', 'REWIRE', 'GROW', 'PRUNE', 'FINAL'];

  const getEntryCategory = (entry: HistoryEntry) => {
    if (entry.label === 'INITIAL' || entry.event_type === 'LEVEL_STARTED') return 'INITIAL';
    if (entry.label === 'FINAL' || entry.event_type === 'LEVEL_COMPLETED') return 'FINAL';
    if (entry.mutation_status === 'ROLLED_BACK' || entry.event_type === 'ROLLBACK' || entry.label?.includes('ROLLED BACK')) return 'ROLLBACK';
    if (entry.mutation_type) return entry.mutation_type.toUpperCase();
    return 'OTHER';
  };

  const categorizedEntries = useMemo(() => {
    const map: Record<string, HistoryEntry[]> = {};
    categories.forEach(c => map[c] = []);
    historyEntries.forEach(entry => {
      const cat = getEntryCategory(entry);
      if (map[cat]) {
        map[cat].push(entry);
      }
    });
    return map;
  }, [historyEntries]);

  // Find the selected entry for the historical indicator
  const selectedEntry = useMemo(() => {
    return historyEntries.find(e => e.history_id === selectedHistoryId);
  }, [historyEntries, selectedHistoryId]);

  const selectedLevelIndex = levelsHistory.find(l => l.level_id === selectedLevelId)?.level_index;

  const handleCategoryClick = (cat: string) => {
    setSelectedCategory(cat);
    const entries = categorizedEntries[cat];
    if (cat === 'INITIAL' && entries && entries.length > 0) {
      selectHistoryEntry(entries[0].history_id);
    } else if (cat === 'FINAL' && entries && entries.length > 0) {
      selectHistoryEntry(entries[0].history_id);
    }
  };

  return (
    <div className="w-full h-full bg-[#060a14] p-5 pt-3 flex flex-col font-mono relative overflow-y-auto overflow-x-hidden">
      
      {/* ─── TITLE & GO LIVE ─── */}
      <div className="flex justify-between items-start mb-2 pb-3 flex-none">
        <h2 className="text-[10px] text-slate-500 tracking-[0.2em] uppercase m-0">
          Level & Structural History
        </h2>
        <button 
          onClick={isHistoricalView ? returnToLive : undefined}
          disabled={!isHistoricalView}
          className={`text-[9px] uppercase tracking-widest transition-colors ${
            isHistoricalView 
              ? 'text-slate-400 hover:text-slate-200 underline decoration-slate-600 underline-offset-4 cursor-pointer'
              : 'text-slate-700 cursor-default'
          }`}
        >
          GO LIVE
        </button>
      </div>

      {/* ─── DROPDOWN ─── */}
      <div className="flex-none mb-4">
        <select
          className="w-full bg-[#02050a] border border-[#1a2744] text-slate-300 text-[10px] py-1 px-2 rounded outline-none cursor-pointer uppercase tracking-widest"
          value={selectedLevelId || ''}
          onChange={(e) => {
            if (e.target.value) {
              selectLevel(e.target.value);
            }
          }}
        >
          <option value="" disabled>SELECT LEVEL</option>
          {Array.from(new Map(levelsHistory.map(lvl => [lvl.level_id, lvl])).values()).sort((a, b) => a.level_index - b.level_index).map(lvl => (
            <option key={lvl.level_id} value={lvl.level_id}>
              LEVEL {String(lvl.level_index + 1).padStart(2, '0')}
            </option>
          ))}
        </select>
      </div>

      {/* ─── CATEGORIES ─── */}
      {selectedLevelId && !historyLoading && (
        <div className="flex-none mb-4 flex flex-col w-full gap-[3px]">
          <div className="flex flex-row justify-between w-full gap-[3px]">
            {categories.slice(0, 4).map(cat => {
              const isActive = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  onClick={() => handleCategoryClick(cat)}
                  className={`text-[9px] flex-1 text-center py-1 border rounded transition-colors tracking-widest ${
                    isActive 
                      ? 'bg-blue-600/30 border-blue-400/50 text-blue-300'
                      : 'bg-[#0a1121] border-[#1a2744] text-slate-500 hover:border-slate-600 hover:text-slate-300'
                  }`}
                >
                  {cat}
                </button>
              );
            })}
          </div>
          <div className="flex flex-row justify-between w-full gap-[3px]">
            {categories.slice(4).map(cat => {
              const isActive = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  onClick={() => handleCategoryClick(cat)}
                  className={`text-[9px] flex-1 text-center py-1 border rounded transition-colors tracking-widest ${
                    isActive 
                      ? 'bg-blue-600/30 border-blue-400/50 text-blue-300'
                      : 'bg-[#0a1121] border-[#1a2744] text-slate-500 hover:border-slate-600 hover:text-slate-300'
                  }`}
                >
                  {cat}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── OCCURRENCES & INDICATOR ─── */}
      <div className="flex-1 flex flex-col gap-4">
        {!selectedLevelId ? (
          <div className="text-slate-600 text-[10px] text-center mt-2">Select a level</div>
        ) : historyLoading ? (
          <div className="text-slate-600 text-[10px] text-center mt-2 animate-pulse">Loading...</div>
        ) : selectedCategory ? (
          <div className="flex flex-col gap-2">
            {categorizedEntries[selectedCategory].length === 0 ? (
              <div className="text-slate-500 text-[10px]">
                No {selectedCategory} occurred at this level.
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {categorizedEntries[selectedCategory].map((entry, idx) => {
                  const isSelected = entry.history_id === selectedHistoryId;
                  const isSingle = (selectedCategory === 'INITIAL' || selectedCategory === 'FINAL') && categorizedEntries[selectedCategory].length === 1;
                  const displayName = isSingle ? selectedCategory : `${selectedCategory} ${idx + 1}`;
                  
                  return (
                    <button
                      key={entry.history_id}
                      onClick={() => selectHistoryEntry(entry.history_id)}
                      className={`text-[9px] px-3 py-1.5 border rounded transition-colors tracking-widest ${
                        isSelected
                          ? 'bg-violet-600/30 border-violet-400/50 text-violet-200'
                          : 'bg-[#0a1121] border-[#1a2744] text-slate-400 hover:border-slate-500 hover:text-slate-200'
                      }`}
                    >
                      {displayName}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        ) : null}

        {/* ─── HISTORICAL INDICATOR ─── */}
        {isHistoricalView && selectedEntry && selectedLevelIndex !== undefined && (
          <div className="mt-auto pt-4 flex flex-col gap-2">
            <div className="text-[9px] text-violet-400 tracking-widest uppercase">
              {(() => {
                const isSingle = (selectedCategory === 'INITIAL' || selectedCategory === 'FINAL') && categorizedEntries[selectedCategory].length === 1;
                const idx = categorizedEntries[selectedCategory || '']?.findIndex(e => e.history_id === selectedHistoryId) + 1;
                const displaySuffix = isSingle ? selectedCategory : `${selectedCategory} ${idx}`;
                return `HISTORICAL · LEVEL ${String(selectedLevelIndex + 1).padStart(2, '0')} · ${displaySuffix}`;
              })()}
            </div>
          </div>
        )}
      </div>
      
    </div>
  );
};
