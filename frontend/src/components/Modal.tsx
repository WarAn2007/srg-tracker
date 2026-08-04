import type { ReactNode } from "react";
import { useEffect, useRef } from "react";

export default function Modal({ title, eyebrow, onClose, children, wide = false }: { title: string; eyebrow: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={`modal-card ${wide ? "modal-wide" : ""}`} role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <header className="modal-header">
          <div><p>{eyebrow}</p><h2 id="modal-title">{title}</h2></div>
          <button ref={closeRef} className="icon-button" onClick={onClose} aria-label={`Close ${title}`}>×</button>
        </header>
        <div className="modal-body">{children}</div>
      </section>
    </div>
  );
}
